// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/autopilot.hpp"
#include "xaos/formulae.hpp"
#include "xaos/renderer.hpp"
#include "qt_executor.hpp"
#include <QApplication>
#include <QAction>
#include <QActionGroup>
#include <QMenu>
#include <QKeySequence>
#include <QCheckBox>
#include <QColor>
#include <QClipboard>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QElapsedTimer>
#include <QEventPoint>
#include <QFileDialog>
#include <QFormLayout>
#include <QFrame>
#include <QHBoxLayout>
#include <QImage>
#include <QGestureEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMenuBar>
#include <QMessageBox>
#include <QMouseEvent>
#include <QNativeGestureEvent>
#include <QPainter>
#include <QPinchGesture>
#include <QPolygonF>
#include <QResizeEvent>
#include <QSpinBox>
#include <QStatusBar>
#include <QTimer>
#include <QToolBar>
#include <QToolButton>
#include <QTouchEvent>
#include <QTransform>
#include <QWheelEvent>
#ifdef Q_OS_ANDROID
#include <QTiltSensor>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <charconv>
#include <condition_variable>
#include <functional>
#include <optional>
#include <numbers>
#include <thread>
#include <vector>

using namespace xaos;
namespace {
/// Parses a complete integer value from a Qt string.
template<class T> T readInteger(const QString&text) {
    const auto s=text.trimmed().toStdString(); T n{};
    auto [p,e]=std::from_chars(s.data(),s.data()+s.size(),n);
    if(e!=std::errc{} || p!=s.data()+s.size()) throw std::invalid_argument("invalid integer");
    return n;
}
/// Returns a small presentation pool that leaves most CPUs available for orbit work.
size_t presentationWorkerCount() noexcept {
    const size_t cpus=defaultWorkerCount();
    return cpus>=8?2:1;
}

/// Wraps immutable presentation storage in a QImage without another framebuffer copy.
QImage makeImage(std::shared_ptr<const DisplayFrame> frame) {
    if(!frame || frame->pixels.empty()) throw std::bad_alloc();
    auto* owner=new std::shared_ptr<const DisplayFrame>(std::move(frame));
    const auto&f=**owner;
    QImage image(reinterpret_cast<const uchar*>(f.pixels.data()),f.request.width,f.request.height,
                 static_cast<qsizetype>(f.request.width)*static_cast<qsizetype>(sizeof(uint32_t)),
                 QImage::Format_ARGB32_Premultiplied,
                 [](void* info){delete static_cast<std::shared_ptr<const DisplayFrame>*>(info);},
                 owner);
    if(image.isNull()) {
        delete owner;
        throw std::bad_alloc();
    }
    return image;
}
class Canvas final:public QWidget {
    struct Job { Request request; size_t threads; uint64_t serial,epoch; bool interactive; };
    struct PresentationJob {
        std::shared_ptr<const FrameBase> frame;
        uint64_t serial,epoch;
        int paletteShift=0;
    };
    std::mutex mutex_;
    std::condition_variable_any wake_;
    std::optional<Job> pending_;
    std::shared_ptr<Cancellation> active_;
    std::atomic<bool> shutdown_{false};
    std::thread coordinator_;
    std::mutex presentationMutex_;
    std::condition_variable_any presentationWake_;
    std::optional<PresentationJob> presentationPending_;
    std::shared_ptr<Cancellation> presentationActive_;
    std::thread presenter_;
    uint64_t serial_=0,shown_=0,epoch_=1;
    uint64_t latestFrameSerial_=0,latestFrameEpoch_=0;
    QImage image_,fallback_;
    View imageView_,fallbackView_;
    QTimer motion_,idle_,autopilotTimer_,touchMomentum_,paletteTimer_;
    QElapsedTimer motionClock_,autopilotClock_,touchSampleClock_,touchMomentumClock_,touchGestureClock_,paletteClock_;
    Autopilot autopilotEngine_;
    std::shared_ptr<const FrameBase> latestFrame_;
    std::shared_ptr<const DisplayFrame> latestDisplay_;
    Statistics latestDisplayStats_;
    std::atomic<int> presentationPaletteShift_{0};
    double autopilotStep_=0,zoomSpeedScale_=1.0;
    double paletteSpeed_=48.0,palettePhase_=0;
    int paletteDirection_=0;
    bool autopilotEnabled_=false;
    QPointF pointer_{.5,.5},lastDrag_;
    int direction_=0;
    bool dragging_=false;
    bool mobileUi_=false;
    bool nativeGestureActive_=false,gestureChanged_=false;
    enum class TouchMode { None, Pan, Pinch };
    TouchMode touchMode_=TouchMode::None;
    QPointF touchLastCenter_,touchStart_,lastTapPosition_,touchLastTwoCenter_;
    QPointF touchPanVelocity_,touchMomentumAnchor_;
    double touchLastDistance_=0,touchLastAngle_=0,touchRotationCandidate_=0;
    double touchZoomVelocity_=0,touchRotationVelocity_=0,touchGestureMaxTravel_=0;
    int touchGestureMaxPoints_=0,touchPointId0_=-1,touchPointId1_=-1;
    bool touchChanged_=false,touchTapCandidate_=false,touchPanStarted_=false;
    bool touchAfterPinchSingle_=false,touchStoppedMotion_=false,touchRotationActive_=false;
#ifdef Q_OS_ANDROID
    QTiltSensor tiltSensor_;
    bool tiltSteeringAvailable_=false,tiltSteeringEnabled_=true;
    bool tiltSteeringFlight_=false,tiltSteeringInverted_=false;
#endif
    QElapsedTimer lastTapClock_;
    size_t threads_=std::max<size_t>(1,defaultWorkerCount()-presentationWorkerCount());
    /// Stops Frax-style kinetic touch motion. Returns whether anything was moving.
    bool stopTouchMomentum(bool refine=true) {
        const bool was=touchMomentum_.isActive();
        touchMomentum_.stop();
        touchPanVelocity_=QPointF{};
        touchZoomVelocity_=0;
        touchRotationVelocity_=0;
#ifdef Q_OS_ANDROID
        tiltSteeringFlight_=false;
#endif
        if(was && refine) submit(false);
        return was;
    }

    /// Blends the newest touch delta into release velocities for kinetic motion.
    void sampleTouchVelocity(const QPointF&panDelta,double zoomLog,double rotation) {
        if(!touchSampleClock_.isValid()) {touchSampleClock_.restart();return;}
        const double seconds=std::clamp(touchSampleClock_.restart()/1000.0,.004,.08);
        constexpr double mix=.55;
        QPointF pan=panDelta/seconds;
        const double speed=std::hypot(pan.x(),pan.y());
        if(speed>5000) pan*=5000.0/speed;
        touchPanVelocity_=touchPanVelocity_*(1.0-mix)+pan*mix;
        const double zoom=std::clamp(zoomLog/seconds,-5.0,5.0);
        const double turn=std::clamp(rotation/seconds,-6.0,6.0);
        touchZoomVelocity_=touchZoomVelocity_*(1.0-mix)+zoom*mix;
        touchRotationVelocity_=touchRotationVelocity_*(1.0-mix)+turn*mix;
    }

    /// Reports whether release velocity is large enough to continue flying.
    bool hasTouchMomentum() const noexcept {
        return std::hypot(touchPanVelocity_.x(),touchPanVelocity_.y())>12.0 ||
               std::abs(touchZoomVelocity_)>.018 || std::abs(touchRotationVelocity_)>.018;
    }

    /// Starts kinetic continuation of the combined pan/zoom/rotation gesture.
    void startTouchMomentum(const QPointF&anchor) {
        if(!hasTouchMomentum()) {stopTouchMomentum(false);idle_.start();return;}
        touchMomentumAnchor_=anchor;
#ifdef Q_OS_ANDROID
        // Frax-style tilt is relative to the phone angle at the instant a
        // panning flight begins. Zoom velocity remains independent of tilt.
        tiltSteeringFlight_=tiltSteeringEnabled_ && tiltSteeringAvailable_ &&
            std::hypot(touchPanVelocity_.x(),touchPanVelocity_.y())>12.0;
        if(tiltSteeringFlight_) tiltSensor_.calibrate();
#endif
        touchMomentumClock_.restart();
        touchMomentum_.start();
    }

    /// Steers Frax-style free motion from relative phone tilt without changing zoom.
    void applyTiltSteering(double seconds) {
#ifdef Q_OS_ANDROID
        if(!tiltSteeringFlight_ || !tiltSteeringEnabled_ || !tiltSteeringAvailable_)
            return;
        const auto*reading=tiltSensor_.reading();
        if(!reading) return;
        auto dead=[](double degrees) {
            constexpr double zone=.8;
            const double magnitude=std::abs(degrees);
            return magnitude<=zone?0.0:std::copysign(magnitude-zone,degrees);
        };
        double tx=dead(reading->xRotation());
        double ty=dead(reading->yRotation());
        if(tiltSteeringInverted_) {tx=-tx;ty=-ty;}

        // A few degrees should be enough to stop/reverse an ordinary throw.
        // Preserve the last zoom velocity exactly, as Frax Motion does.
        constexpr double panAcceleration=300.0; // logical pixels/s^2 per degree
        touchPanVelocity_+=QPointF(-tx,ty)*(panAcceleration*seconds);
        const double speed=std::hypot(touchPanVelocity_.x(),touchPanVelocity_.y());
        if(speed>5000.0) touchPanVelocity_*=5000.0/speed;

        // Tilt also steers an already-spinning flight, but never creates rotation
        // from a pure pan. This avoids accidental grid invalidation.
        if(std::abs(touchRotationVelocity_)>.018)
            touchRotationVelocity_=std::clamp(
                touchRotationVelocity_-tx*.02*seconds,-6.0,6.0);
#else
        (void)seconds;
#endif
    }

    /// Frax tap zoom: exact 3x step and move the tapped mathematical point to center.
    void fraxTapZoom(const QPointF&position,double factor) {
        const int w=std::max(1,width()),h=std::max(1,height());
        view.zoom(position.x()/w,position.y()/h,factor,w,h);
        const QPointF center(width()*.5,height()*.5);
        view.pan(center.x()-position.x(),center.y()-position.y(),w);
        pointer_=center;
        submit(true);
    }

    /// Maps a target selected in a displayed source frame into the current viewport.
    QPointF mapDisplayFocus(const DisplayFrame&frame,double u,double v) const {
        if(width()<1 || height()<1) return QPointF(.5,.5);
        const auto point=frame.request.view.screenToComplex(
            u,v,std::max(1,frame.request.width),std::max(1,frame.request.height));
        const auto current=view.complexToScreen(
            point.first,point.second,std::max(1,width()),std::max(1,height()));
        return QPointF(std::clamp(current.first,0.0,1.0)*width(),
                       std::clamp(current.second,0.0,1.0)*height());
    }

    /// Restores the current formula's XaoS default view and parameter seed.
    void restoreFormulaDefault() {
        const auto&info=formulaInfo(settings.formula);
        const mp_bitcnt_t p=std::max<mp_bitcnt_t>(128,settings.minimumPrecision);
        const double aspect=static_cast<double>(std::max(1,height()))/std::max(1,width());
        const double span=std::max(info.horizontalSpan,info.verticalSpan/aspect);
        view={Big::fromDouble(info.centerRe,p),Big::fromDouble(info.centerIm,p),
              Big::fromDouble(span,p)};
        settings.juliaRe=Big::fromDouble(info.seedRe,p);
        settings.juliaIm=Big::fromDouble(info.seedIm,p);
        autopilotEngine_.reset();autopilotStep_=0;latestDisplay_.reset();
    }

    /// Applies XaoS's accelerated zoom/unzoom step selected by the autopilot.
    void autopilotTick() {
        if(!autopilotEnabled_ || !latestDisplay_) return;
        const double seconds=std::clamp<double>(autopilotClock_.restart()/1000.0,.001,.2);
        auto decision=autopilotEngine_.tick(*latestDisplay_,latestDisplayStats_.complete,
                                            latestDisplay_->request.view.span,1);
        if(decision.control==AutopilotControl::Reset) {
            restoreFormulaDefault();
            submit(true);
            return;
        }

        pointer_=mapDisplayFocus(*latestDisplay_,decision.focusX,decision.focusY);
        const double speedup=.0018*zoomSpeedScale_; // original STEP times user speed
        const double maximum=.024*zoomSpeedScale_;  // original MAXSTEP times user speed
        const double mul=seconds/.05;    // original FRAMERATE=20 time multiplier

        if(decision.control==AutopilotControl::ZoomIn)
            autopilotStep_=std::min(maximum,autopilotStep_+speedup*2*mul);
        else if(decision.control==AutopilotControl::ZoomOut)
            autopilotStep_=std::max(-maximum,autopilotStep_-speedup*2*mul);
        else if(autopilotStep_>0)
            autopilotStep_=std::max(0.0,autopilotStep_-speedup*mul);
        else if(autopilotStep_<0)
            autopilotStep_=std::min(0.0,autopilotStep_+speedup*mul);

        if(autopilotStep_==0) return;
        try {
            const double factor=std::pow(1.0-autopilotStep_,mul);
            view.zoom(pointer_.x()/std::max(1,width()),pointer_.y()/std::max(1,height()),
                      factor,std::max(1,width()),std::max(1,height()));
            submit(true);
        } catch(const std::exception&e) {
            setAutopilot(false);
            if(onStatus) onStatus(QString("Autopilot stopped: ")+e.what());
        }
    }

    /// Queues the newest computed grid for asynchronous presentation.
    void queuePresentation(std::shared_ptr<const FrameBase> frame,uint64_t serial,uint64_t epoch) {
        {
            std::lock_guard stateLock(mutex_);
            if(epoch!=epoch_) return; // semantic settings changed while the grid was computing
            latestFrame_=frame;
            latestFrameSerial_=serial;latestFrameEpoch_=epoch;
        }
        const int paletteShift=presentationPaletteShift_.load(std::memory_order_relaxed);
        {
            std::lock_guard lock(presentationMutex_);
            // Geometry-only motion intentionally does not invalidate an older
            // completed grid: drawView() can transform that source image into the
            // current viewport, exactly as classic XaoS keeps showing/refining
            // frames while the zoom target moves. Keep only the newest pending grid.
            presentationPending_=PresentationJob{std::move(frame),serial,epoch,paletteShift};
        }
        presentationWake_.notify_one();
    }

    /// Recolors the newest mathematical grid without submitting any orbit/DP work.
    void queuePalettePresentation() {
        std::shared_ptr<const FrameBase> frame;
        uint64_t serial=0,epoch=0;
        {
            std::lock_guard lock(mutex_);
            frame=latestFrame_;
            serial=latestFrameSerial_;
            epoch=latestFrameEpoch_;
        }
        if(!frame) {
            // During startup there may not be a grid yet; one ordinary request is
            // enough to seed presentation-only color cycling.
            submit(true);
            return;
        }
        const int paletteShift=presentationPaletteShift_.load(std::memory_order_relaxed);
        {
            std::lock_guard lock(presentationMutex_);
            if(presentationActive_)
                presentationActive_->cancelled.store(true,std::memory_order_relaxed);
            presentationPending_=PresentationJob{std::move(frame),serial,epoch,paletteShift};
        }
        presentationWake_.notify_one();
        update();
    }

    /// Reconstructs display frames independently from the compute coordinator.
    void presentationLoop() {
        ThreadExecutor executor(presentationWorkerCount());
        std::shared_ptr<const DisplayFrame> previous;
        while(!shutdown_.load(std::memory_order_relaxed)) {
            PresentationJob job;
            std::shared_ptr<Cancellation> token;
            {
                std::unique_lock lock(presentationMutex_);
                presentationWake_.wait(lock,[&]{
                    return shutdown_.load(std::memory_order_relaxed) || presentationPending_.has_value();
                });
                if(shutdown_.load(std::memory_order_relaxed)) return;
                job=std::move(*presentationPending_);
                presentationPending_.reset();
                token=std::make_shared<Cancellation>();
                presentationActive_=token;
            }
            try {
                auto display=presentFrame(*job.frame,executor,*token,previous.get(),job.paletteShift);
                if(token->cancelled.load(std::memory_order_relaxed)) {
                    std::lock_guard lock(presentationMutex_);
                    if(presentationActive_==token) presentationActive_.reset();
                    continue;
                }
                auto image=makeImage(display);
                if(token->cancelled.load(std::memory_order_relaxed)) {
                    std::lock_guard lock(presentationMutex_);
                    if(presentationActive_==token) presentationActive_.reset();
                    continue;
                }
                previous=display;
                const auto stats=job.frame->stats;
                const auto view=job.frame->request.view;
                const auto reconstruction=job.frame->request.settings.reconstruction;
                const bool savedState=job.frame->request.settings.saveState;
                const double presentationMs=display->milliseconds;
                QMetaObject::invokeMethod(this,
                    [this,image=std::move(image),display,view,stats,reconstruction,savedState,presentationMs,
                     id=job.serial,epoch=job.epoch] {
                        if(epoch!=epoch_ || id<shown_) return;
                        shown_=id;
                        ++publishedFrames;
                        fallback_=image_; fallbackView_=imageView_;
                        image_=image; imageView_=view;
                        latestDisplay_=display;latestDisplayStats_=stats;
                        if(stats.complete) ++completedFrames;
                        const char* mode="nearest";
                        switch(reconstruction) {
                        case Reconstruction::Nearest: mode="nearest"; break;
                        case Reconstruction::Bilinear: mode="bilinear"; break;
                        case Reconstruction::Bicubic: mode="bicubic"; break;
                        }
                        if(onStatus) onStatus(
                            QString("%1%2 | %3 bits | %4 | compute %5 ms | present %6 ms | reused %7 resumed %8 | %9 | %10%11")
                                .arg(QString::fromStdString(stats.backend)).arg(stats.simd?" / AVX2":"")
                                .arg(static_cast<qulonglong>(stats.bits))
                                .arg(savedState?"state":"counts")
                                .arg(stats.milliseconds,0,'f',1).arg(presentationMs,0,'f',1)
                                .arg(static_cast<qulonglong>(stats.reused))
                                .arg(static_cast<qulonglong>(stats.resumed))
                                .arg(QString::fromLatin1(mode))
                                .arg(stats.uniform?"uniform samples":"adaptive preview")
                                .arg(stats.complete?QString{}:QString(" / refining (guess %1, fill %2)")
                                    .arg(static_cast<qulonglong>(stats.solidGuessed))
                                    .arg(static_cast<qulonglong>(stats.filled))));
                        update();
                    },Qt::QueuedConnection);
            } catch(const std::exception&e) {
                if(token->cancelled.load(std::memory_order_relaxed)) continue;
                const QString message=QString::fromUtf8(e.what());
                QMetaObject::invokeMethod(this,[this,message,id=job.serial,epoch=job.epoch] {
                    if(epoch==epoch_ && id>=shown_ && onStatus) onStatus("Presentation error: "+message);
                },Qt::QueuedConnection);
            }
            std::lock_guard lock(presentationMutex_);
            if(presentationActive_==token) presentationActive_.reset();
        }
    }

    /// Runs the compute loop, coalescing requests and immediately scheduling further refinement.
    void coordinator() {
        struct DynamicBudget {
            std::array<double,50> calculation{};
            size_t pos=0,count=0;
            /// Returns the recent average compute time.
            double average(double fallback) const {
                if(!count) return fallback;
                double sum=0; for(size_t i=0;i<count;++i) sum+=calculation[i];
                return sum/static_cast<double>(count);
            }
            /// Chooses the next orbit/refinement budget without charging presentation time.
            unsigned next(bool interactive) const {
                const double calc=average(40.0);
                double ms=calc*5.0;
                if(interactive) {
                    if(ms>1000.0/25.0) ms=calc*3.0;
                    ms=std::min(ms,1000.0/15.0);
                } else ms=1000.0/3.0;
                ms=std::max(ms,1000.0/30.0);
                return static_cast<unsigned>(std::lround(std::max(ms,10.0)));
            }
            /// Records only mathematical compute/refinement time.
            void observe(double calc) {
                calculation[pos]=calc; pos=(pos+1)%calculation.size();
                count=std::min(calculation.size(),count+1);
            }
        } budget;
        Renderer renderer;
        std::unique_ptr<QtExecutor> executor;
        while(!shutdown_.load(std::memory_order_relaxed)) {
            Job job;
            std::shared_ptr<Cancellation> token;
            {
                std::unique_lock lock(mutex_);
                wake_.wait(lock,[&]{
                    return shutdown_.load(std::memory_order_relaxed) || pending_.has_value();
                });
                if(shutdown_.load(std::memory_order_relaxed)) return;
                job=std::move(*pending_); pending_.reset();
                token=std::make_shared<Cancellation>();
                job.request.settings.sliceMilliseconds=budget.next(job.interactive);
                active_=token;
            }
            try {
                if(!executor || executor->concurrency()!=job.threads)
                    executor=std::make_unique<QtExecutor>(job.threads);
                auto frame=renderer.render(job.request,*executor,*token);
                budget.observe(frame->stats.milliseconds);
                queuePresentation(frame,job.serial,job.epoch);
                std::lock_guard lock(mutex_);
                if(active_==token) active_.reset();
                // Compute refinement continues immediately; it no longer waits for
                // interpolation, framebuffer copies, or Qt image publication.
                if(!frame->stats.complete && !pending_ &&
                   !shutdown_.load(std::memory_order_relaxed))
                    pending_=job;
                if(pending_) wake_.notify_one();
            } catch(const std::exception&e) {
                const QString message=QString::fromUtf8(e.what());
                QMetaObject::invokeMethod(this,[this,message,id=job.serial,epoch=job.epoch] {
                    if(epoch==epoch_ && id>=shown_ && onStatus) onStatus("Render error: "+message);
                },Qt::QueuedConnection);
                std::lock_guard lock(mutex_);
                if(active_==token) active_.reset();
            }
        }
    }
    /// Draws a cached image through the affine transform between two rotated viewports.
    void drawView(QPainter&p,const QImage&image,const View&source) {
        if(image.isNull() || width()<1||height()<1) return;
        auto targetPoint=[&](double u,double v) {
            const auto point=source.screenToComplex(u,v,image.width(),image.height());
            const auto uv=view.complexToScreen(point.first,point.second,width(),height());
            return QPointF(uv.first*width(),uv.second*height());
        };
        const QPolygonF sourceQuad{
            QPointF(0,0),QPointF(image.width(),0),
            QPointF(image.width(),image.height()),QPointF(0,image.height())};
        const QPolygonF targetQuad{
            targetPoint(0,0),targetPoint(1,0),targetPoint(1,1),targetPoint(0,1)};
        QTransform transform;
        if(!QTransform::quadToQuad(sourceQuad,targetQuad,transform)) return;
        bool invertible=false;
        const QTransform inverse=transform.inverted(&invertible);
        if(!invertible) return;

        // Determine which source-space rectangle covers the current widget. When
        // zooming out, extend the nearest boundary row/column exactly as before;
        // doing it before the affine transform also works for rotated viewports.
        const QPolygonF widgetQuad{
            QPointF(0,0),QPointF(width(),0),QPointF(width(),height()),QPointF(0,height())};
        QRectF needed=inverse.map(widgetQuad).boundingRect();
        if(!std::isfinite(needed.left())||!std::isfinite(needed.right())||
           !std::isfinite(needed.top())||!std::isfinite(needed.bottom()))
            return;
        const double iw=image.width(),ih=image.height();
        const double left=std::min(0.0,needed.left());
        const double right=std::max(iw,needed.right());
        const double top=std::min(0.0,needed.top());
        const double bottom=std::max(ih,needed.bottom());

        p.save();
        p.setTransform(transform,true);
        if(left<0)
            p.drawImage(QRectF(left,0,-left,ih),image,QRectF(0,0,1,ih));
        if(right>iw)
            p.drawImage(QRectF(iw,0,right-iw,ih),image,QRectF(iw-1,0,1,ih));
        if(top<0)
            p.drawImage(QRectF(0,top,iw,-top),image,QRectF(0,0,iw,1));
        if(bottom>ih)
            p.drawImage(QRectF(0,ih,iw,bottom-ih),image,QRectF(0,ih-1,iw,1));
        if(left<0 && top<0)
            p.drawImage(QRectF(left,top,-left,-top),image,QRectF(0,0,1,1));
        if(right>iw && top<0)
            p.drawImage(QRectF(iw,top,right-iw,-top),image,QRectF(iw-1,0,1,1));
        if(left<0 && bottom>ih)
            p.drawImage(QRectF(left,ih,-left,bottom-ih),image,QRectF(0,ih-1,1,1));
        if(right>iw && bottom>ih)
            p.drawImage(QRectF(iw,ih,right-iw,bottom-ih),image,QRectF(iw-1,ih-1,1,1));
        p.drawImage(QRectF(0,0,iw,ih),image);
        p.restore();
    }
protected:
    /// Re-render adaptively after a phone orientation/window-size change while
    /// immediately continuing to draw the transformed previous image.
    void resizeEvent(QResizeEvent*event) override {
        const QSize old=event->oldSize();
        QWidget::resizeEvent(event);
        if(old.width()>0 && old.height()>0 && width()>0 && height()>0) {
            const double widthRatio=static_cast<double>(width())/old.width();
            pointer_.setX(pointer_.x()*widthRatio);
            pointer_.setY(pointer_.y()*static_cast<double>(height())/old.height());

            // On phones a portrait/landscape rotation should preserve fractal
            // scale, not preserve horizontal field of view. Keeping span/width
            // constant leaves the overlapping old/new viewport on the same
            // row/column lattice, so exact samples survive the orientation change.
            if(mobileUi_ && publishedFrames>0)
                view.span=scale(view.span,widthRatio);
        }
#ifdef Q_OS_ANDROID
        if(tiltSteeringFlight_) tiltSensor_.calibrate();
#endif
        if(publishedFrames>0) submit(true);
    }

    /// Handles direct mobile touch plus native trackpad and generic pinch gestures.
    bool event(QEvent*event) override {
        if(mobileUi_ && (event->type()==QEvent::TouchBegin ||
                         event->type()==QEvent::TouchUpdate ||
                         event->type()==QEvent::TouchEnd ||
                         event->type()==QEvent::TouchCancel)) {
            auto*touch=static_cast<QTouchEvent*>(event);

            // Qt does not promise touch-point list order. Keep the pair ordered by
            // stable point id so a list-order swap cannot look like a 180-degree twist.
            std::vector<std::pair<int,QPointF>> orderedActive;
            orderedActive.reserve(2);
            for(const auto&point:touch->points()) {
                if(point.state()!=QEventPoint::State::Released && orderedActive.size()<2)
                    orderedActive.emplace_back(point.id(),point.position());
            }
            std::sort(orderedActive.begin(),orderedActive.end(),
                      [](const auto&a,const auto&b){return a.first<b.first;});
            std::vector<QPointF> active;
            active.reserve(orderedActive.size());
            for(const auto&point:orderedActive) active.push_back(point.second);
            const int activeId0=orderedActive.empty()?-1:orderedActive[0].first;
            const int activeId1=orderedActive.size()<2?-1:orderedActive[1].first;

            if(event->type()==QEvent::TouchBegin) {
                const bool stoppedAutopilot=autopilotEnabled_;
                if(stoppedAutopilot) setAutopilot(false);
                touchStoppedMotion_=stopTouchMomentum(false) || stoppedAutopilot;
                idle_.stop();
                touchGestureClock_.restart();
                touchSampleClock_.restart();
                touchGestureMaxPoints_=static_cast<int>(active.size());
                touchGestureMaxTravel_=0;
                touchChanged_=false;
                touchTapCandidate_=active.size()==1;
                touchPanStarted_=false;
                touchAfterPinchSingle_=false;
                touchPanVelocity_=QPointF{};
                touchZoomVelocity_=0;
                touchRotationVelocity_=0;
                touchRotationCandidate_=0;
                touchRotationActive_=false;
                touchPointId0_=touchPointId1_=-1;
                if(!active.empty()) {
                    touchStart_=touchLastCenter_=active.front();
                    pointer_=active.front();
                }
            }

            touchGestureMaxPoints_=std::max(touchGestureMaxPoints_,
                static_cast<int>(touch->points().size()));
            for(const auto&point:touch->points()) {
                const QPointF travel=point.position()-point.pressPosition();
                touchGestureMaxTravel_=std::max(touchGestureMaxTravel_,
                    std::hypot(travel.x(),travel.y()));
            }
            if(touch->points().size()>=2) {
                touchLastTwoCenter_=(touch->points()[0].position()+touch->points()[1].position())*.5;
            }

            const QPointF releasedPosition=touch->points().empty()
                ? touchLastCenter_ : touch->points().front().position();
            if(event->type()==QEvent::TouchEnd || event->type()==QEvent::TouchCancel ||
               active.empty()) {
                const bool cancelled=event->type()==QEvent::TouchCancel;
                const bool quick=touchGestureClock_.isValid() && touchGestureClock_.elapsed()<380;
                const bool twoFingerTap=!cancelled && !touchStoppedMotion_ &&
                    touchGestureMaxPoints_>=2 && quick && touchGestureMaxTravel_<14.0 &&
                    !touchRotationActive_ && !touchChanged_;
                bool handledTap=false;

                try {
                    if(twoFingerTap) {
                        // Frax/Maps convention: two-finger tap is an exact 3x step out.
                        stopTouchMomentum(false);
                        fraxTapZoom(touchLastTwoCenter_,3.0);
                        lastTapClock_.invalidate();
                        handledTap=true;
                    } else if(!cancelled && !touchStoppedMotion_ &&
                              touchGestureMaxPoints_==1 && touchTapCandidate_ &&
                              !touchPanStarted_) {
                        const bool doubleTap=lastTapClock_.isValid() &&
                            lastTapClock_.elapsed()<350 &&
                            std::hypot(releasedPosition.x()-lastTapPosition_.x(),
                                       releasedPosition.y()-lastTapPosition_.y())<48.0;
                        if(doubleTap) {
                            // Frax tap zoom is a discrete 300% step and recenters the target.
                            stopTouchMomentum(false);
                            fraxTapZoom(releasedPosition,1.0/3.0);
                            lastTapClock_.invalidate();
                            handledTap=true;
                        } else {
                            lastTapPosition_=releasedPosition;
                            lastTapClock_.restart();
                            // A plain tap stops flight and asks for the best still image.
                            submit(false);
                        }
                    }
                } catch(const std::exception&ex) {
                    stopTouchMomentum(false);
                    if(onStatus) onStatus(ex.what());
                }

                if(!handledTap && touchChanged_) {
                    submit(true);
                    if(!cancelled && hasTouchMomentum())
                        startTouchMomentum(pointer_);
                    else {
                        stopTouchMomentum(false);
                        idle_.start();
                    }
                } else if(!handledTap && touchStoppedMotion_) {
                    // Frax single-tap semantics: stopping animation is itself the action.
                    lastTapClock_.invalidate();
                    submit(false);
                } else if(!handledTap && !touchChanged_) {
                    idle_.start();
                }

                touchMode_=TouchMode::None;
                touchChanged_=false;
                touchTapCandidate_=false;
                touchPanStarted_=false;
                touchAfterPinchSingle_=false;
                touchStoppedMotion_=false;
                touchGestureMaxPoints_=0;
                touchGestureMaxTravel_=0;
                touchRotationCandidate_=0;
                touchRotationActive_=false;
                touchPointId0_=touchPointId1_=-1;
                dragging_=false;
                touch->accept();return true;
            }

            idle_.stop();
            try {
                if(active.size()>=2) {
                    const QPointF center=(active[0]+active[1])*.5;
                    const QPointF separation=active[1]-active[0];
                    const double distance=std::hypot(separation.x(),separation.y());
                    const double angle=std::atan2(separation.y(),separation.x());
                    touchTapCandidate_=false;
                    touchAfterPinchSingle_=false;
                    dragging_=false;

                    const bool newPair=touchMode_!=TouchMode::Pinch ||
                        activeId0!=touchPointId0_ || activeId1!=touchPointId1_;
                    if(newPair) {
                        // Re-baseline when the second finger arrives or the active pair
                        // changes. Besides preventing pinch-as-pan, this avoids both a
                        // distance spike and an angle jump from touch-point replacement.
                        touchMode_=TouchMode::Pinch;
                        touchPointId0_=activeId0;touchPointId1_=activeId1;
                        touchLastCenter_=center;
                        touchLastDistance_=distance;
                        touchLastAngle_=angle;
                        touchRotationCandidate_=0;
                        touchRotationActive_=false;
                        touchPanVelocity_=QPointF{};
                        touchZoomVelocity_=0;
                        touchRotationVelocity_=0;
                        touchSampleClock_.restart();
                    } else {
                        bool changed=false;
                        const QPointF delta=center-touchLastCenter_;
                        double zoomLog=0,rotation=0;
                        if(std::hypot(delta.x(),delta.y())>.01) {
                            view.pan(delta.x(),delta.y(),std::max(1,width()));
                            changed=true;
                        }
                        if(distance>4.0 && touchLastDistance_>4.0) {
                            const double scale=distance/touchLastDistance_;
                            if(std::isfinite(scale) && scale>0 && std::abs(scale-1.0)>1e-4) {
                                zoomLog=std::log(scale);
                                view.zoom(center.x()/std::max(1,width()),
                                          center.y()/std::max(1,height()),1.0/scale,
                                          std::max(1,width()),std::max(1,height()));
                                changed=true;
                            }
                        }
                        const double rawRotation=
                            std::remainder(angle-touchLastAngle_,2.0*std::numbers::pi);
                        // Rotation is exceptionally noisy when the fingers are close.
                        // More importantly, *any* accidental angle change invalidates
                        // XaoS's reusable row/column coordinate system. Keep rotation
                        // locked until the user has made a deliberate twist.
                        const double minSeparation=std::clamp(width()*.09,32.0,56.0);
                        constexpr double unlockRotation=6.0*std::numbers::pi/180.0;
                        constexpr double maxSampleRotation=35.0*std::numbers::pi/180.0;
                        const bool stableAngle=distance>=minSeparation &&
                            touchLastDistance_>=minSeparation &&
                            std::abs(rawRotation)<=maxSampleRotation;
                        if(!touchRotationActive_) {
                            if(stableAngle) {
                                touchRotationCandidate_+=rawRotation;
                                if(std::abs(touchRotationCandidate_)>=unlockRotation) {
                                    // Unlock without applying the dead-zone angle. The
                                    // next sample starts smooth rotation with no snap.
                                    touchRotationActive_=true;
                                    touchRotationCandidate_=0;
                                    touchRotationVelocity_=0;
                                }
                            } else {
                                touchRotationCandidate_=0;
                            }
                            rotation=0;
                        } else if(stableAngle) {
                            rotation=rawRotation;
                        } else {
                            // A one-frame tracking glitch should never rotate the
                            // image by tens of degrees and destroy the reusable grid.
                            rotation=0;
                            touchRotationVelocity_=0;
                        }
                        if(std::abs(rotation)>1e-5) {
                            view.rotate(center.x()/std::max(1,width()),
                                        center.y()/std::max(1,height()),rotation,
                                        std::max(1,width()),std::max(1,height()));
                            changed=true;
                        }
                        if(changed) sampleTouchVelocity(delta,zoomLog,rotation);
                        touchLastCenter_=center;
                        touchLastDistance_=distance;
                        touchLastAngle_=angle;
                        pointer_=center;
                        if(changed) {touchChanged_=true;submit(true);}
                    }
                } else {
                    const QPointF position=active.front();
                    pointer_=position;

                    if(touchMode_==TouchMode::Pinch) {
                        // Fingers normally lift one after another. Keep the pinch
                        // velocity alive during this short tail so release inertia is
                        // based on the two-finger gesture, not a synthetic one-finger pan.
                        if(!touchAfterPinchSingle_) {
                            touchAfterPinchSingle_=true;
                            touchStart_=touchLastCenter_=position;
                        } else if(std::hypot(position.x()-touchStart_.x(),
                                             position.y()-touchStart_.y())>=8.0) {
                            touchMode_=TouchMode::Pan;
                            touchPanStarted_=true;
                            touchPanVelocity_=QPointF{};
                            touchZoomVelocity_=0;
                            touchRotationVelocity_=0;
                            touchLastCenter_=position;
                            touchSampleClock_.restart();
                        }
                    } else if(touchMode_!=TouchMode::Pan) {
                        touchMode_=TouchMode::Pan;
                        touchStart_=touchLastCenter_=position;
                        touchTapCandidate_=true;
                        touchPanStarted_=false;
                        touchSampleClock_.restart();
                    } else if(!touchPanStarted_) {
                        // The dead zone gives a second finger time to land without
                        // moving the image underneath an intended pinch.
                        if(std::hypot(position.x()-touchStart_.x(),
                                      position.y()-touchStart_.y())>=8.0) {
                            touchPanStarted_=true;
                            touchTapCandidate_=false;
                            touchLastCenter_=position;
                            touchSampleClock_.restart();
                        }
                    } else {
                        const QPointF delta=position-touchLastCenter_;
                        touchLastCenter_=position;
                        if(std::hypot(delta.x(),delta.y())>.01) {
                            view.pan(delta.x(),delta.y(),std::max(1,width()));
                            sampleTouchVelocity(delta,0,0);
                            touchChanged_=true;
                            submit(true);
                        }
                    }
                }
            } catch(const std::exception&ex) {
                stopTouchMomentum(false);
                if(onStatus) onStatus(ex.what());
            }
            touch->accept();return true;
        }
        if(event->type()==QEvent::NativeGesture) {
            auto*gesture=static_cast<QNativeGestureEvent*>(event);
            if(autopilotEnabled_) {gesture->accept();return true;}
            const auto type=gesture->gestureType();
            if(type==Qt::BeginNativeGesture) {
                nativeGestureActive_=true;gestureChanged_=false;idle_.stop();
                gesture->accept();return true;
            }
            if(type==Qt::EndNativeGesture) {
                nativeGestureActive_=false;
                if(gestureChanged_) submit(true); else update();
                gestureChanged_=false;gesture->accept();return true;
            }
            const QPointF position=gesture->position();
            const double u=position.x()/std::max(1,width());
            const double v=position.y()/std::max(1,height());
            try {
                if(type==Qt::RotateNativeGesture) {
                    view.rotate(u,v,gesture->value()*std::numbers::pi/180.0,
                                std::max(1,width()),std::max(1,height()));
                    gestureChanged_=true;
                } else if(type==Qt::ZoomNativeGesture) {
                    const double magnification=1.0+gesture->value();
                    if(magnification>0) {
                        view.zoom(u,v,1.0/magnification,
                                  std::max(1,width()),std::max(1,height()));
                        gestureChanged_=true;
                    }
                } else if(type==Qt::PanNativeGesture) {
                    const QPointF delta=gesture->delta();
                    view.pan(delta.x(),delta.y(),std::max(1,width()));
                    gestureChanged_=true;
                } else return QWidget::event(event);
                pointer_=position;idle_.stop();update();
            } catch(const std::exception&ex) {
                if(onStatus) onStatus(ex.what());
            }
            gesture->accept();return true;
        }
        if(event->type()==QEvent::Gesture && !nativeGestureActive_) {
            auto*gestureEvent=static_cast<QGestureEvent*>(event);
            if(auto*pinch=static_cast<QPinchGesture*>(gestureEvent->gesture(Qt::PinchGesture))) {
                if(autopilotEnabled_) {gestureEvent->accept(pinch);return true;}
                if(pinch->state()==Qt::GestureStarted) {
                    gestureChanged_=false;idle_.stop();
                }
                // The generic Qt pinch recognizer stores center points in global
                // coordinates; macOS uses its specialized native recognizer and
                // reports them in widget coordinates.
                QPointF center=pinch->centerPoint();
                QPointF lastCenter=pinch->lastCenterPoint();
#ifndef Q_OS_MACOS
                center=QPointF(mapFromGlobal(center.toPoint()));
                lastCenter=QPointF(mapFromGlobal(lastCenter.toPoint()));
#endif
                const double u=center.x()/std::max(1,width());
                const double v=center.y()/std::max(1,height());
                try {
                    const auto flags=pinch->changeFlags();
                    if(flags.testFlag(QPinchGesture::CenterPointChanged)) {
                        const QPointF delta=center-lastCenter;
                        view.pan(delta.x(),delta.y(),std::max(1,width()));
                        gestureChanged_=true;
                    }
                    if(flags.testFlag(QPinchGesture::ScaleFactorChanged) && pinch->scaleFactor()>0) {
                        view.zoom(u,v,1.0/pinch->scaleFactor(),
                                  std::max(1,width()),std::max(1,height()));
                        gestureChanged_=true;
                    }
                    if(flags.testFlag(QPinchGesture::RotationAngleChanged)) {
                        const double degrees=pinch->rotationAngle()-pinch->lastRotationAngle();
                        view.rotate(u,v,degrees*std::numbers::pi/180.0,
                                    std::max(1,width()),std::max(1,height()));
                        gestureChanged_=true;
                    }
                    pointer_=center;update();
                } catch(const std::exception&ex) {
                    if(onStatus) onStatus(ex.what());
                }
                if(pinch->state()==Qt::GestureFinished || pinch->state()==Qt::GestureCanceled) {
                    if(gestureChanged_) submit(true);
                    gestureChanged_=false;
                }
                gestureEvent->accept(pinch);return true;
            }
        }
        return QWidget::event(event);
    }
    /// Paints the current and fallback fractal images plus the interaction hint.
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        // QPainter backends differ in their default image-scaling behavior. XaoS
        // is a sample-reuse zoomer, not a bilinear image zoomer: keep moved rows
        // and columns crisp while new resolution is being computed.
        p.setRenderHint(QPainter::SmoothPixmapTransform,false);
        p.fillRect(rect(),Qt::black);
        drawView(p,fallback_,fallbackView_);drawView(p,image_,imageView_);
        if(!mobileUi_) {
            p.setPen(Qt::white);
            p.drawText(12,22,"Hold left/right: zoom   |   Middle drag: pan   |   Wheel/pinch: zoom   |   Two-finger twist: rotate   |   A: autopilot");
        }
    }
    /// Starts zooming or panning in response to a mouse press.
    void mousePressEvent(QMouseEvent*e) override {
        if(mobileUi_ && e->source()!=Qt::MouseEventNotSynthesized) {e->accept();return;}
        if(autopilotEnabled_) {e->accept();return;}
        pointer_=e->position();
        if(mobileUi_ && e->button()==Qt::LeftButton) {
            dragging_=true;lastDrag_=pointer_;e->accept();return;
        }
        if(e->button()==Qt::MiddleButton) {dragging_=true;lastDrag_=pointer_;}
        else if(e->button()==Qt::LeftButton || e->button()==Qt::RightButton) {
            direction_=e->button()==Qt::LeftButton?1:-1;motionClock_.restart();motion_.start();
        }
    }
    /// Stops the active mouse interaction and schedules idle refinement.
    void mouseReleaseEvent(QMouseEvent*e) override {
        if(mobileUi_ && e->source()!=Qt::MouseEventNotSynthesized) {e->accept();return;}
        if(autopilotEnabled_) {e->accept();return;}
        if(e->button()==Qt::MiddleButton || (mobileUi_ && e->button()==Qt::LeftButton)) {
            dragging_=false;idle_.start();
        }
        if(!mobileUi_ && (e->button()==Qt::LeftButton || e->button()==Qt::RightButton)) {
            direction_=0;motion_.stop();idle_.start();
        }
    }
    /// Updates the zoom focus or pans while the middle button is held.
    void mouseMoveEvent(QMouseEvent*e) override {
        if(mobileUi_ && e->source()!=Qt::MouseEventNotSynthesized) {e->accept();return;}
        if(autopilotEnabled_) {e->accept();return;}
        pointer_=e->position();
        if(dragging_) {
            auto d=pointer_-lastDrag_;lastDrag_=pointer_;
            try {view.pan(d.x(),d.y(),std::max(1,width()));submit(true);} catch(const std::exception&ex){if(onStatus)onStatus(ex.what());}
        }
    }
    /// Double-tap/double-click zooms into the touched point in the mobile UI.
    void mouseDoubleClickEvent(QMouseEvent*e) override {
        if(mobileUi_ && e->source()!=Qt::MouseEventNotSynthesized) {e->accept();return;}
        if(!mobileUi_ || autopilotEnabled_ || e->button()!=Qt::LeftButton) {
            QWidget::mouseDoubleClickEvent(e);return;
        }
        pointer_=e->position();
        try {
            stopTouchMomentum(false);
            fraxTapZoom(pointer_,1.0/3.0);
        } catch(const std::exception&ex) { if(onStatus) onStatus(ex.what()); }
        e->accept();
    }
    /// Applies a stepped pointer-centred zoom from the mouse wheel.
    void wheelEvent(QWheelEvent*e) override {
        if(autopilotEnabled_) {e->accept();return;}
        const double steps=e->angleDelta().y()/120.;
        try {
            view.zoom(e->position().x()/std::max(1,width()),e->position().y()/std::max(1,height()),
                      std::exp(-.2*steps),std::max(1,width()),std::max(1,height()));submit(true);
        }catch(const std::exception&ex){if(onStatus)onStatus(ex.what());}
        e->accept();
    }
public:
    View view;
    Settings settings;
    int completedFrames=0;
    int publishedFrames=0;
    std::function<void(QString)> onStatus;
    std::function<void(bool)> onAutopilotChanged;
    std::function<void()> onColorChanged;
    /// Constructs a Canvas instance.
    explicit Canvas(QWidget*parent=nullptr):QWidget(parent) {
        setMouseTracking(true);setFocusPolicy(Qt::StrongFocus);
        setAttribute(Qt::WA_AcceptTouchEvents,true);
        grabGesture(Qt::PinchGesture);
        motion_.setInterval(16);idle_.setSingleShot(true);idle_.setInterval(180);
        touchMomentum_.setInterval(16);
        paletteTimer_.setInterval(40); // classic XaoS color cycling cadence
        autopilotTimer_.setInterval(40); // original XaoS autopilot timer: 25 Hz
        connect(&touchMomentum_,&QTimer::timeout,this,[this] {
            const double seconds=std::clamp(touchMomentumClock_.restart()/1000.0,.001,.05);
            try {
                bool changed=false;
                applyTiltSteering(seconds);
                const QPointF pan=touchPanVelocity_*seconds;
                if(std::hypot(pan.x(),pan.y())>.01) {
                    view.pan(pan.x(),pan.y(),std::max(1,width()));
                    touchMomentumAnchor_+=pan;
                    changed=true;
                }
                if(std::abs(touchZoomVelocity_)>.001) {
                    view.zoom(touchMomentumAnchor_.x()/std::max(1,width()),
                              touchMomentumAnchor_.y()/std::max(1,height()),
                              std::exp(-touchZoomVelocity_*seconds),
                              std::max(1,width()),std::max(1,height()));
                    changed=true;
                }
                if(std::abs(touchRotationVelocity_)>.001) {
                    view.rotate(touchMomentumAnchor_.x()/std::max(1,width()),
                                touchMomentumAnchor_.y()/std::max(1,height()),
                                touchRotationVelocity_*seconds,
                                std::max(1,width()),std::max(1,height()));
                    changed=true;
                }
                pointer_=touchMomentumAnchor_;
                if(changed) submit(true);

                // Frax Motion keeps the release velocity until the user taps
                // to stop it. This is deliberately not inertial scrolling friction:
                // a thrown pan/pinch/twist is an animation state in its own right.
            } catch(const std::exception&e) {
                stopTouchMomentum(false);
                if(onStatus) onStatus(e.what());
            }
        });
        connect(&motion_,&QTimer::timeout,this,[this] {
            if(!direction_) return;
            const double seconds=std::min<qint64>(motionClock_.restart(),100)/1000.;
            try {
                view.zoom(pointer_.x()/std::max(1,width()),pointer_.y()/std::max(1,height()),
                          std::exp(-direction_*seconds*.75*zoomSpeedScale_),
                          std::max(1,width()),std::max(1,height()));submit(true);
            }catch(const std::exception&e){motion_.stop();if(onStatus)onStatus(e.what());}
        });
        connect(&idle_,&QTimer::timeout,this,[this]{submit(false);});
        connect(&paletteTimer_,&QTimer::timeout,this,[this] {
            if(!paletteDirection_) return;
            const double seconds=std::clamp(paletteClock_.restart()/1000.0,.001,.2);
            palettePhase_+=static_cast<double>(paletteDirection_)*paletteSpeed_*seconds;
            int delta=palettePhase_>0?static_cast<int>(std::floor(palettePhase_)):
                                      static_cast<int>(std::ceil(palettePhase_));
            if(!delta) return;
            palettePhase_-=delta;
            const int period=static_cast<int>(std::max<size_t>(1,classicDefaultPalette().size()-1));
            int shift=(settings.paletteShift+delta)%period;
            if(shift<0) shift+=period;
            settings.paletteShift=shift;
            presentationPaletteShift_.store(shift,std::memory_order_relaxed);
            queuePalettePresentation();
        });
        connect(&autopilotTimer_,&QTimer::timeout,this,[this]{autopilotTick();});
        presenter_=std::thread([this]{presentationLoop();});
        coordinator_=std::thread([this]{coordinator();});
    }
    /// Releases resources owned by the Canvas instance.
    ~Canvas() override {
        motion_.stop();idle_.stop();autopilotTimer_.stop();touchMomentum_.stop();paletteTimer_.stop();
        shutdown_.store(true,std::memory_order_relaxed);
        {
            std::lock_guard lock(mutex_);
            pending_.reset();
            if(active_) active_->cancelled.store(true,std::memory_order_relaxed);
        }
        {
            std::lock_guard lock(presentationMutex_);
            presentationPending_.reset();
            if(presentationActive_)
                presentationActive_->cancelled.store(true,std::memory_order_relaxed);
        }
        wake_.notify_all();
        presentationWake_.notify_all();
        if(coordinator_.joinable()) coordinator_.join();
        if(presenter_.joinable()) presenter_.join();
    }
    /// Queues the newest request, coalescing viewport motion without starving active refinement.
    void submit(bool interactive=false,bool invalidate=false) {
        if(width()<1||height()<1) return;
        const double dpr=devicePixelRatioF();
        Request request{view,std::max(1,static_cast<int>(std::ceil(width()*dpr))),
                             std::max(1,static_cast<int>(std::ceil(height()*dpr))),settings};
        request.settings.uniform=false;
        request.settings.focusX=pointer_.x()/width();request.settings.focusY=pointer_.y()/height();

        uint64_t serial=0,epoch=0;
        {
            std::lock_guard lock(mutex_);
            serial=++serial_;
            if(invalidate) {
                ++epoch_;
                autopilotEngine_.reset();
                autopilotStep_=0;
                latestFrame_.reset();
                latestFrameSerial_=latestFrameEpoch_=0;
                latestDisplay_.reset();
            }
            epoch=epoch_;

            // Continuous zoom/pan only replaces the pending target. Let the active
            // bounded slice reach its normal line/time boundary, otherwise a 16 ms
            // mouse timer can repeatedly kill the very row/column work that should
            // become the next visible refinement.
            if(invalidate && active_)
                active_->cancelled.store(true,std::memory_order_relaxed);
            pending_=Job{std::move(request),threads_,serial,epoch,interactive};

            if(invalidate) {
                std::lock_guard presentationLock(presentationMutex_);
                if(presentationActive_)
                    presentationActive_->cancelled.store(true,std::memory_order_relaxed);
                presentationPending_.reset();
            }
        }
        wake_.notify_one();update();
        if(interactive) idle_.start();
    }
    /// Selects a formula, restores its XaoS default view/seed, and invalidates old state.
    void setFormula(Formula formula) {
        settings.formula=formula;
        restoreFormulaDefault();
        submit(false,true);
    }

    /// Enables or disables the XaoS-style automatic fractal explorer.
    void setAutopilot(bool enabled) {
        if(autopilotEnabled_==enabled) return;
        autopilotEnabled_=enabled;
        direction_=0;motion_.stop();stopTouchMomentum(false);dragging_=false;
        if(onAutopilotChanged) onAutopilotChanged(enabled);
        autopilotEngine_.reset();autopilotStep_=0;
        if(enabled) {
            autopilotClock_.restart();
            autopilotTimer_.start();
            if(onStatus) onStatus("Autopilot: searching fractal boundaries");
        } else {
            autopilotTimer_.stop();
            idle_.start();
            if(onStatus) onStatus("Autopilot off");
        }
    }
    /// Reports whether automatic fractal exploration is enabled.
    bool autopilotEnabled() const noexcept { return autopilotEnabled_; }

    /// Starts/stops XaoS-style palette cycling. Direction is -1, 0, or +1.
    void setPaletteCycling(int direction) {
        direction=std::clamp(direction,-1,1);
        if(paletteDirection_==direction) return;
        paletteDirection_=direction;
        palettePhase_=0;
        if(direction) {
            paletteClock_.restart();
            paletteTimer_.start();
        } else {
            paletteTimer_.stop();
            submit(false);
        }
        if(onColorChanged) onColorChanged();
        if(onStatus) onStatus(direction>0?"Palette cycling forward":
                              direction<0?"Palette cycling backward":"Palette cycling stopped");
    }
    int paletteCyclingDirection() const noexcept { return paletteDirection_; }

    /// Changes XaoS color-cycling speed without touching mathematical state.
    void adjustPaletteSpeed(bool faster) {
        const double factor=1.25;
        paletteSpeed_=std::clamp(faster?paletteSpeed_*factor:paletteSpeed_/factor,1.0,4096.0);
        if(onStatus) onStatus(QString("Palette cycling: %1 entries/s").arg(paletteSpeed_,0,'f',1));
    }

    /// Shifts the palette without recalculating any orbit.
    void shiftPalette(int delta) {
        const int period=static_cast<int>(std::max<size_t>(1,classicDefaultPalette().size()-1));
        int shift=(settings.paletteShift+delta)%period;
        if(shift<0) shift+=period;
        settings.paletteShift=shift;
        presentationPaletteShift_.store(shift,std::memory_order_relaxed);
        queuePalettePresentation();
        if(onColorChanged) onColorChanged();
    }

    void resetPaletteShift() {
        settings.paletteShift=0;
        presentationPaletteShift_.store(0,std::memory_order_relaxed);
        queuePalettePresentation();
        if(onColorChanged) onColorChanged();
    }

    void setInColoring(InColoring mode) {
        if(settings.inColoring==mode) return;
        settings.inColoring=mode;
        submit(false);
        if(onColorChanged) onColorChanged();
    }
    void setOutColoring(OutColoring mode) {
        if(settings.outColoring==mode) return;
        settings.outColoring=mode;
        submit(false);
        if(onColorChanged) onColorChanged();
    }

    /// Adjusts XaoS zoom acceleration/max-step by the historical 1.05 factor.
    void adjustZoomSpeed(bool faster) {
        constexpr double factor=1.05;
        zoomSpeedScale_=std::clamp(faster?zoomSpeedScale_*factor:zoomSpeedScale_/factor,
                                   1.0/1024.0,1024.0);
        if(onStatus) onStatus(QString("Zoom speed: %1x").arg(zoomSpeedScale_,0,'f',3));
    }
    /// Returns the current zoom speed multiplier relative to XaoS defaults.
    double zoomSpeedScale() const noexcept { return zoomSpeedScale_; }

    /// Switches interaction hints and one-finger behavior for the phone layout.
    void setMobileUi(bool enabled) {
        if(mobileUi_==enabled) return;
        mobileUi_=enabled;
        // The phone UI handles QTouchEvent directly so a one-finger synthetic
        // mouse drag cannot win the race against Qt's pinch recognizer.
        if(enabled) ungrabGesture(Qt::PinchGesture);
        else grabGesture(Qt::PinchGesture);
#ifdef Q_OS_ANDROID
        if(enabled) {
            tiltSensor_.setDataRate(60);
            // Feature metadata is valid only after a backend is connected.
            tiltSteeringAvailable_=tiltSensor_.connectToBackend();
            if(tiltSteeringAvailable_ &&
               tiltSensor_.isFeatureSupported(QSensor::AxesOrientation))
                tiltSensor_.setAxesOrientationMode(QSensor::AutomaticOrientation);
            if(tiltSteeringAvailable_)
                tiltSteeringAvailable_=tiltSensor_.start();
        } else {
            tiltSensor_.stop();
            tiltSteeringAvailable_=false;
            tiltSteeringFlight_=false;
        }
#endif
        update();
    }
    /// Returns whether phone-oriented interaction is enabled.
    bool mobileUi() const noexcept { return mobileUi_; }

    bool tiltSteeringAvailable() const noexcept {
#ifdef Q_OS_ANDROID
        return tiltSteeringAvailable_;
#else
        return false;
#endif
    }
    bool tiltSteeringEnabled() const noexcept {
#ifdef Q_OS_ANDROID
        return tiltSteeringEnabled_;
#else
        return false;
#endif
    }
    void setTiltSteering(bool enabled) {
#ifdef Q_OS_ANDROID
        tiltSteeringEnabled_=enabled;
        if(!enabled) tiltSteeringFlight_=false;
        if(onStatus) onStatus(enabled?"Tilt steering on":"Tilt steering off");
#else
        (void)enabled;
#endif
    }
    bool tiltSteeringInverted() const noexcept {
#ifdef Q_OS_ANDROID
        return tiltSteeringInverted_;
#else
        return false;
#endif
    }
    void setTiltSteeringInverted(bool inverted) {
#ifdef Q_OS_ANDROID
        tiltSteeringInverted_=inverted;
#else
        (void)inverted;
#endif
    }
    /// Returns the number of compute workers, excluding presentation workers.
    size_t workerCount() const noexcept { return threads_; }
    /// Changes the worker count and requests a new render.
    void setThreads(size_t n) {threads_=n;submit();}
    /// Restores the default fractal view and requests a render.
    void reset() {stopTouchMomentum(false);restoreFormulaDefault();submit();}
    /// Stops continuous zooming and requests refinement of the current view.
    void stopZoom() {direction_=0;motion_.stop();stopTouchMomentum(false);setAutopilot(false);submit();}
    /// Writes the currently displayed Qt image to a user-selected PNG file.
    void saveImage() {
        if(image_.isNull()) return;
        const auto path=QFileDialog::getSaveFileName(this,"Save displayed frame",{},"PNG (*.png)");
        if(!path.isEmpty() && !image_.save(path)) QMessageBox::warning(this,"Save failed","Could not write the image.");
    }
    /// Edits arbitrary-precision view and Julia parameters in a dialog.
    void coordinates() {
        QDialog d(this);d.setWindowTitle("Arbitrary-precision view");
        QFormLayout form(&d);
        QLineEdit re,im,span;
        for(auto*field:{&re,&im,&span}) field->setMaxLength(std::numeric_limits<int>::max());
        re.setText(QString::fromStdString(view.re.str()));im.setText(QString::fromStdString(view.im.str()));span.setText(QString::fromStdString(view.span.str()));
        QLineEdit rotation(QString::number(view.rotation*180.0/std::numbers::pi,'g',15));
        QLineEdit precision(QString::number(static_cast<qulonglong>(settings.minimumPrecision)));
        QLineEdit jr,ji;
        jr.setMaxLength(std::numeric_limits<int>::max());ji.setMaxLength(std::numeric_limits<int>::max());
        jr.setText(QString::fromStdString(settings.juliaRe.str()));ji.setText(QString::fromStdString(settings.juliaIm.str()));
        form.addRow("Center, real",&re);form.addRow("Center, imaginary",&im);form.addRow("Horizontal span",&span);
        form.addRow("Rotation (degrees)",&rotation);
        form.addRow("Minimum bits (0 = adaptive)",&precision);form.addRow("Julia c, real",&jr);form.addRow("Julia c, imaginary",&ji);
        QLabel note("Precision grows with zoom depth. More bits invalidate old orbits.\nMemory and CPU time remain finite; direct GMP is not perturbation rendering.");
        form.addRow(&note);
        QDialogButtonBox buttons(QDialogButtonBox::Ok|QDialogButtonBox::Cancel);
        form.addRow(&buttons);connect(&buttons,&QDialogButtonBox::accepted,&d,&QDialog::accept);connect(&buttons,&QDialogButtonBox::rejected,&d,&QDialog::reject);
        d.resize(650,d.sizeHint().height());
        if(d.exec()==QDialog::Accepted) {
            try {
                View next=View::parse(re.text().trimmed().toStdString(),im.text().trimmed().toStdString(),span.text().trimmed().toStdString(),std::max(1,width()),16,settings.memoryBudget);
                bool rotationOk=false;
                const double degrees=rotation.text().trimmed().toDouble(&rotationOk);
                if(!rotationOk || !std::isfinite(degrees)) throw std::invalid_argument("invalid rotation");
                next.rotation=View::normalizeRotation(degrees*std::numbers::pi/180.0);
                auto bits=readInteger<mp_bitcnt_t>(precision.text());
                const auto r=jr.text().trimmed().toStdString(),i=ji.text().trimmed().toStdString();
                Big real=Big::parse(r,View::textBits(r,i,"")),imag=Big::parse(i,View::textBits(r,i,""));
                view=std::move(next);settings.minimumPrecision=bits;settings.juliaRe=std::move(real);settings.juliaIm=std::move(imag);submit(false,true);
            }catch(const std::exception&e){QMessageBox::warning(this,"Invalid view",e.what());}
        }
    }
};
class Window final:public QMainWindow {
    bool mobile_=false;
    QSpinBox*iterations_=nullptr;
    QLabel*mobileBadge_=nullptr;
    QFrame*mobileDock_=nullptr;
    QToolButton*mobileExplore_=nullptr;
    QToolButton*mobileFormula_=nullptr;
    QToolButton*mobileDetail_=nullptr;
    QToolButton*mobileQuality_=nullptr;
    QMenu*mobileFormulaMenu_=nullptr;
    QMenu*mobileColorMenu_=nullptr;
    QMenu*mobileMoreMenu_=nullptr;

    QString qualityName() const {
        switch(canvas->settings.reconstruction) {
        case Reconstruction::Nearest:return "CRISP";
        case Reconstruction::Bilinear:return "LINEAR";
        case Reconstruction::Bicubic:return "CUBIC";
        }
        return "CRISP";
    }
    void refreshMobileChrome() {
        if(!mobile_) return;
        const auto&info=formulaInfo(canvas->settings.formula);
        mobileBadge_->setText(QString(" XaoS 30  ·  %1 ").arg(QString::fromLatin1(info.name)));
        mobileFormula_->setText(QString::fromLatin1(info.shortName).toUpper()+"\nFORMULA");
        mobileDetail_->setText(QString::number(canvas->settings.iterations)+"\nDETAIL");
        if(canvas->paletteCyclingDirection())
            mobileQuality_->setText(canvas->paletteCyclingDirection()>0?"CYCLE >\nCOLOR":"< CYCLE\nCOLOR");
        else
            mobileQuality_->setText(QString::fromLatin1(outColoringName(canvas->settings.outColoring))
                                    .left(8).toUpper()+"\nCOLOR");
        mobileExplore_->setText(canvas->autopilotEnabled()?"STOP\nEXPLORE":"EXPLORE\nAUTO");
        mobileExplore_->setChecked(canvas->autopilotEnabled());
        mobileBadge_->adjustSize();
        layoutMobileChrome();
    }
    void layoutMobileChrome() {
        if(!mobile_ || !mobileDock_ || !mobileBadge_) return;
        const int w=canvas->width(),h=canvas->height();
        const QPoint origin=canvas->mapTo(this,QPoint(0,0));
        const int margin=std::clamp(w/28,12,22);
        const int dockHeight=std::clamp(h/10,68,88);
        mobileDock_->setGeometry(origin.x()+margin,origin.y()+h-dockHeight-margin,
                                 std::max(120,w-2*margin),dockHeight);
        mobileBadge_->adjustSize();
        mobileBadge_->move(origin.x()+margin,origin.y()+margin);
        mobileDock_->raise();mobileBadge_->raise();
    }
    QToolButton* mobileButton(const QString&text,QWidget*parent) {
        auto*b=new QToolButton(parent);
        // Keep buttons on Qt's normal touch-to-mouse path. Canvas handles raw
        // QTouchEvent only for the fractal surface.
        b->setAttribute(Qt::WA_AcceptTouchEvents,false);
        b->setText(text);
        b->setToolButtonStyle(Qt::ToolButtonTextOnly);
        b->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Expanding);
        b->setMinimumWidth(46);
        b->setCursor(Qt::PointingHandCursor);
        return b;
    }
    void buildMobileUi() {
        canvas->setMobileUi(true);
        menuBar()->hide();statusBar()->hide();
        for(auto*bar:findChildren<QToolBar*>()) bar->hide();

        // Mobile chrome must be a sibling of the touch canvas, not its child.
        // Otherwise Qt walks an unhandled button touch up to Canvas (the first
        // WA_AcceptTouchEvents ancestor), and Canvas consumes it as a pan/tap.
        mobileBadge_=new QLabel(this);
        mobileBadge_->setAttribute(Qt::WA_StyledBackground,true);
        mobileBadge_->setAttribute(Qt::WA_TransparentForMouseEvents,true);
        mobileBadge_->setStyleSheet(
            "QLabel{color:white;background:rgba(8,10,16,190);"
            "border:1px solid rgba(255,255,255,36);border-radius:16px;"
            "padding:7px 12px;font-size:14px;font-weight:650;}");

        mobileDock_=new QFrame(this);
        mobileDock_->setAttribute(Qt::WA_StyledBackground,true);
        mobileDock_->setStyleSheet(
            "QFrame{background:rgba(8,10,16,214);border:1px solid rgba(255,255,255,38);"
            "border-radius:24px;}"
            "QToolButton{color:rgba(255,255,255,225);background:transparent;border:0;"
            "border-radius:15px;padding:7px 5px;font-size:10px;font-weight:650;}"
            "QToolButton:pressed{background:rgba(92,220,255,42);}"
            "QToolButton:checked{color:rgb(102,232,255);background:rgba(92,220,255,34);}"
            "QMenu{color:white;background:rgb(20,22,30);border:1px solid rgb(55,60,72);"
            "padding:8px;font-size:15px;}QMenu::item{padding:11px 24px;border-radius:8px;}"
            "QMenu::item:selected{background:rgb(45,74,86);}");

        auto*layout=new QHBoxLayout(mobileDock_);
        layout->setContentsMargins(8,7,8,7);layout->setSpacing(2);
        mobileExplore_=mobileButton("EXPLORE\nAUTO",mobileDock_);
        mobileExplore_->setCheckable(true);
        mobileFormula_=mobileButton("MANDEL\nFORMULA",mobileDock_);
        mobileDetail_=mobileButton("512\nDETAIL",mobileDock_);
        mobileQuality_=mobileButton("ITER\nCOLOR",mobileDock_);
        auto*reset=mobileButton("RESET\nVIEW",mobileDock_);
        auto*more=mobileButton("MORE\n···",mobileDock_);
        for(auto*b:{mobileExplore_,mobileFormula_,mobileDetail_,mobileQuality_,reset,more})
            layout->addWidget(b);

        mobileFormulaMenu_=new QMenu(mobileFormula_);
        for(const auto&info:formulaInfos()) {
            auto*a=mobileFormulaMenu_->addAction(QString::fromLatin1(info.name));
            const auto formula=info.formula;
            connect(a,&QAction::triggered,this,[this,formula]{
                canvas->setFormula(formula);refreshMobileChrome();
            });
        }
        mobileFormula_->setMenu(mobileFormulaMenu_);
        mobileFormula_->setPopupMode(QToolButton::InstantPopup);

        mobileColorMenu_=new QMenu(mobileQuality_);
        auto*cycleForward=mobileColorMenu_->addAction("Cycle palette forward");
        auto*cycleBackward=mobileColorMenu_->addAction("Cycle palette backward");
        auto*cycleStop=mobileColorMenu_->addAction("Stop palette cycling");
        mobileColorMenu_->addSeparator();
        auto*shiftForward=mobileColorMenu_->addAction("Shift palette +1");
        auto*shiftBackward=mobileColorMenu_->addAction("Shift palette -1");
        auto*shiftReset=mobileColorMenu_->addAction("Reset palette shift");
        auto*cycleFaster=mobileColorMenu_->addAction("Cycle faster");
        auto*cycleSlower=mobileColorMenu_->addAction("Cycle slower");
        mobileColorMenu_->addSeparator();

        auto*outMenu=mobileColorMenu_->addMenu("Outside coloring");
        auto*outGroup=new QActionGroup(outMenu);outGroup->setExclusive(true);
        for(int i=0;i<10;++i) {
            const auto mode=static_cast<OutColoring>(i);
            auto*a=outMenu->addAction(QString::fromLatin1(outColoringName(mode)));
            a->setCheckable(true);a->setChecked(canvas->settings.outColoring==mode);
            a->setData(i);outGroup->addAction(a);
            connect(a,&QAction::triggered,this,[this,mode]{
                canvas->setOutColoring(mode);refreshMobileChrome();
            });
        }

        auto*inMenu=mobileColorMenu_->addMenu("Inside coloring");
        auto*inGroup=new QActionGroup(inMenu);inGroup->setExclusive(true);
        for(int i=0;i<10;++i) {
            const auto mode=static_cast<InColoring>(i);
            auto*a=inMenu->addAction(QString::fromLatin1(inColoringName(mode)));
            a->setCheckable(true);a->setChecked(canvas->settings.inColoring==mode);
            a->setData(i);inGroup->addAction(a);
            connect(a,&QAction::triggered,this,[this,mode]{
                canvas->setInColoring(mode);refreshMobileChrome();
            });
        }

        auto*reconstructMenu=mobileColorMenu_->addMenu("Reconstruction");
        auto*reconstructGroup=new QActionGroup(reconstructMenu);reconstructGroup->setExclusive(true);
        for(auto [name,mode]:std::array<std::pair<const char*,Reconstruction>,3>{{
                {"Nearest (XaoS)",Reconstruction::Nearest},
                {"Bilinear",Reconstruction::Bilinear},
                {"Bicubic",Reconstruction::Bicubic}}}) {
            auto*a=reconstructMenu->addAction(name);
            a->setCheckable(true);a->setChecked(canvas->settings.reconstruction==mode);
            a->setData(static_cast<int>(mode));reconstructGroup->addAction(a);
            connect(a,&QAction::triggered,this,[this,mode]{
                canvas->settings.reconstruction=mode;canvas->submit(false,true);refreshMobileChrome();
            });
        }

        connect(cycleForward,&QAction::triggered,this,[this]{canvas->setPaletteCycling(1);refreshMobileChrome();});
        connect(cycleBackward,&QAction::triggered,this,[this]{canvas->setPaletteCycling(-1);refreshMobileChrome();});
        connect(cycleStop,&QAction::triggered,this,[this]{canvas->setPaletteCycling(0);refreshMobileChrome();});
        connect(shiftForward,&QAction::triggered,canvas,[this]{canvas->shiftPalette(1);});
        connect(shiftBackward,&QAction::triggered,canvas,[this]{canvas->shiftPalette(-1);});
        connect(shiftReset,&QAction::triggered,canvas,[this]{canvas->resetPaletteShift();});
        connect(cycleFaster,&QAction::triggered,canvas,[this]{canvas->adjustPaletteSpeed(true);});
        connect(cycleSlower,&QAction::triggered,canvas,[this]{canvas->adjustPaletteSpeed(false);});
        connect(mobileColorMenu_,&QMenu::aboutToShow,this,[this,outMenu,inMenu,reconstructMenu] {
            for(auto*a:outMenu->actions())
                a->setChecked(a->data().toInt()==static_cast<int>(canvas->settings.outColoring));
            for(auto*a:inMenu->actions())
                a->setChecked(a->data().toInt()==static_cast<int>(canvas->settings.inColoring));
            for(auto*a:reconstructMenu->actions())
                a->setChecked(a->data().toInt()==static_cast<int>(canvas->settings.reconstruction));
        });
        mobileQuality_->setMenu(mobileColorMenu_);
        mobileQuality_->setPopupMode(QToolButton::InstantPopup);

        mobileMoreMenu_=new QMenu(more);
        auto*coordinates=mobileMoreMenu_->addAction("Coordinates & precision");
        auto*tilt=mobileMoreMenu_->addAction("Tilt steering");
        tilt->setCheckable(true);
        tilt->setChecked(canvas->tiltSteeringEnabled());
        tilt->setEnabled(canvas->tiltSteeringAvailable());
        auto*invertTilt=mobileMoreMenu_->addAction("Invert tilt steering");
        invertTilt->setCheckable(true);
        invertTilt->setChecked(canvas->tiltSteeringInverted());
        invertTilt->setEnabled(canvas->tiltSteeringAvailable());
        auto*level=mobileMoreMenu_->addAction("Level rotation");
        auto*saveState=mobileMoreMenu_->addAction("Save orbit state");
        saveState->setCheckable(true);saveState->setChecked(canvas->settings.saveState);
        auto*save=mobileMoreMenu_->addAction("Save frame as PNG");
        mobileMoreMenu_->addSeparator();
        auto*help=mobileMoreMenu_->addAction("Gesture guide");
        more->setMenu(mobileMoreMenu_);more->setPopupMode(QToolButton::InstantPopup);

        connect(mobileExplore_,&QToolButton::clicked,this,[this](bool checked){
            canvas->setAutopilot(checked);refreshMobileChrome();
        });
        connect(mobileDetail_,&QToolButton::clicked,this,[this]{
            static constexpr std::array<uint32_t,7> levels{{128,256,512,1024,2048,4096,8192}};
            auto it=std::upper_bound(levels.begin(),levels.end(),canvas->settings.iterations);
            setIterations(it==levels.end()?levels.front():*it);
        });
        connect(reset,&QToolButton::clicked,canvas,[this]{canvas->reset();refreshMobileChrome();});
        connect(coordinates,&QAction::triggered,canvas,&Canvas::coordinates);
        connect(tilt,&QAction::toggled,canvas,&Canvas::setTiltSteering);
        connect(invertTilt,&QAction::toggled,canvas,&Canvas::setTiltSteeringInverted);
        connect(level,&QAction::triggered,this,[this]{
            const double delta=-canvas->view.rotation;
            if(delta!=0) {
                canvas->view.rotate(.5,.5,delta,std::max(1,canvas->width()),std::max(1,canvas->height()));
                canvas->submit(true);
            }
        });
        connect(saveState,&QAction::toggled,this,[this](bool on){
            canvas->settings.saveState=on;canvas->submit();
        });
        connect(save,&QAction::triggered,canvas,&Canvas::saveImage);
        connect(help,&QAction::triggered,this,[this]{
            QMessageBox::information(this,"Explore XaoS",
                "Motion works like Frax:\n\n"
                "Swipe with one finger to pan; release with speed to coast.\n"
                "Move, pinch and twist two fingers together — pan, zoom and rotation combine.\n"
                "Release a moving gesture to keep flying; tap once to stop and refine.\n"
                "While a pan is flying, tilt the phone a few degrees to steer, stop or reverse it.\n"
                "Tilt is relative to the phone angle at release and does not change zoom speed.\n"
                "Double-tap one finger: exact 3× zoom in and center that point.\n"
                "Tap with two fingers: exact 3× zoom out and center the midpoint.\n\n"
                "Explore lets XaoS choose the next interesting boundary automatically.");
        });
        canvas->onAutopilotChanged=[this](bool){refreshMobileChrome();};
        canvas->onColorChanged=[this]{refreshMobileChrome();};
        canvas->onStatus=[this](const QString&s){
            if(mobileBadge_) mobileBadge_->setToolTip(s);
        };
        refreshMobileChrome();
    }

protected:
    void resizeEvent(QResizeEvent*event) override {
        QMainWindow::resizeEvent(event);
        layoutMobileChrome();
    }

public:
    Canvas*canvas=nullptr;
    /// Constructs a desktop or phone-focused window.
    explicit Window(bool mobile=false):mobile_(mobile) {
        canvas=new Canvas(this);setCentralWidget(canvas);
        setWindowTitle("XaoS 30");
        if(mobile_) {
            buildMobileUi();
            return;
        }

        setWindowTitle("XaoS Modern — reusable orbits / arbitrary precision");
        auto*bar=addToolBar("Rendering");bar->setMovable(false);
        auto*formula=new QComboBox(bar);
        for(const auto&info:formulaInfos())
            formula->addItem(QString::fromLatin1(info.name),static_cast<int>(info.formula));
        bar->addWidget(formula);
        bar->addWidget(new QLabel("  Iterations ",bar));
        iterations_=new QSpinBox(bar);iterations_->setRange(1,2000000000);iterations_->setValue(512);bar->addWidget(iterations_);
        auto*states=new QCheckBox("Save orbits",bar);states->setChecked(true);bar->addWidget(states);
        bar->addWidget(new QLabel("  Reconstruction ",bar));
        auto*reconstruction=new QComboBox(bar);
        reconstruction->addItems({"Nearest (XaoS)","Bilinear","Bicubic"});
        bar->addWidget(reconstruction);
        bar->addWidget(new QLabel("  Workers ",bar));auto*threads=new QSpinBox(bar);threads->setRange(1,1024);
        threads->setValue(static_cast<int>(canvas->workerCount()));bar->addWidget(threads);
        auto*autopilot=bar->addAction("Autopilot");autopilot->setCheckable(true);
        autopilot->setShortcut(QKeySequence(Qt::Key_A));
        auto*coords=bar->addAction("Coordinates / bits");auto*reset=bar->addAction("Reset");
        connect(formula,qOverload<int>(&QComboBox::currentIndexChanged),this,[this,formula](int i){
            canvas->setFormula(static_cast<Formula>(formula->itemData(i).toInt()));
        });
        connect(iterations_,qOverload<int>(&QSpinBox::valueChanged),this,[this](int n){
            canvas->settings.iterations=static_cast<uint32_t>(n);canvas->submit(false,true);
        });
        connect(states,&QCheckBox::toggled,this,[this](bool b){canvas->settings.saveState=b;canvas->submit();});
        connect(reconstruction,qOverload<int>(&QComboBox::currentIndexChanged),this,[this](int i){
            canvas->settings.reconstruction=static_cast<Reconstruction>(i);canvas->submit(false,true);
        });
        connect(threads,qOverload<int>(&QSpinBox::valueChanged),this,[this](int n){canvas->setThreads(static_cast<size_t>(n));});
        connect(autopilot,&QAction::toggled,canvas,&Canvas::setAutopilot);
        canvas->onAutopilotChanged=[autopilot](bool enabled){autopilot->setChecked(enabled);};
        connect(coords,&QAction::triggered,canvas,&Canvas::coordinates);connect(reset,&QAction::triggered,canvas,&Canvas::reset);

        auto*colorMenu=menuBar()->addMenu("Color");
        auto*cycleForward=colorMenu->addAction("Cycle palette forward");
        cycleForward->setShortcut(QKeySequence(Qt::Key_Y));
        auto*cycleBackward=colorMenu->addAction("Cycle palette backward");
        cycleBackward->setShortcut(QKeySequence(Qt::SHIFT|Qt::Key_Y));
        auto*cycleStop=colorMenu->addAction("Stop palette cycling");
        colorMenu->addSeparator();
        auto*shiftForward=colorMenu->addAction("Shift palette +1");
        shiftForward->setShortcut(QKeySequence(Qt::Key_Plus));
        auto*shiftBackward=colorMenu->addAction("Shift palette -1");
        shiftBackward->setShortcut(QKeySequence(Qt::Key_Minus));
        auto*shiftReset=colorMenu->addAction("Reset palette shift");
        auto*cycleFaster=colorMenu->addAction("Cycle faster");
        auto*cycleSlower=colorMenu->addAction("Cycle slower");
        colorMenu->addSeparator();

        auto*outMenu=colorMenu->addMenu("Outside coloring");
        auto*outGroup=new QActionGroup(outMenu);outGroup->setExclusive(true);
        for(int i=0;i<10;++i) {
            const auto mode=static_cast<OutColoring>(i);
            auto*a=outMenu->addAction(QString::fromLatin1(outColoringName(mode)));
            a->setCheckable(true);a->setData(i);a->setChecked(canvas->settings.outColoring==mode);
            outGroup->addAction(a);
            connect(a,&QAction::triggered,canvas,[this,mode]{canvas->setOutColoring(mode);});
        }

        auto*inMenu=colorMenu->addMenu("Inside coloring");
        auto*inGroup=new QActionGroup(inMenu);inGroup->setExclusive(true);
        for(int i=0;i<10;++i) {
            const auto mode=static_cast<InColoring>(i);
            auto*a=inMenu->addAction(QString::fromLatin1(inColoringName(mode)));
            a->setCheckable(true);a->setData(i);a->setChecked(canvas->settings.inColoring==mode);
            inGroup->addAction(a);
            connect(a,&QAction::triggered,canvas,[this,mode]{canvas->setInColoring(mode);});
        }

        connect(cycleForward,&QAction::triggered,canvas,[this]{
            canvas->setPaletteCycling(canvas->paletteCyclingDirection()==1?0:1);
        });
        connect(cycleBackward,&QAction::triggered,canvas,[this]{
            canvas->setPaletteCycling(canvas->paletteCyclingDirection()==-1?0:-1);
        });
        connect(cycleStop,&QAction::triggered,canvas,[this]{canvas->setPaletteCycling(0);});
        connect(shiftForward,&QAction::triggered,canvas,[this]{canvas->shiftPalette(1);});
        connect(shiftBackward,&QAction::triggered,canvas,[this]{canvas->shiftPalette(-1);});
        connect(shiftReset,&QAction::triggered,canvas,[this]{canvas->resetPaletteShift();});
        connect(cycleFaster,&QAction::triggered,canvas,[this]{canvas->adjustPaletteSpeed(true);});
        connect(cycleSlower,&QAction::triggered,canvas,[this]{canvas->adjustPaletteSpeed(false);});

        auto*nextOut=new QAction(this);nextOut->setShortcut(QKeySequence(Qt::Key_C));addAction(nextOut);
        connect(nextOut,&QAction::triggered,canvas,[this]{
            const int next=(static_cast<int>(canvas->settings.outColoring)+1)%10;
            canvas->setOutColoring(static_cast<OutColoring>(next));
        });
        auto*nextIn=new QAction(this);nextIn->setShortcut(QKeySequence(Qt::Key_F));addAction(nextIn);
        connect(nextIn,&QAction::triggered,canvas,[this]{
            const int next=(static_cast<int>(canvas->settings.inColoring)+1)%10;
            canvas->setInColoring(static_cast<InColoring>(next));
        });
        connect(colorMenu,&QMenu::aboutToShow,this,[this,outMenu,inMenu] {
            for(auto*a:outMenu->actions())
                a->setChecked(a->data().toInt()==static_cast<int>(canvas->settings.outColoring));
            for(auto*a:inMenu->actions())
                a->setChecked(a->data().toInt()==static_cast<int>(canvas->settings.inColoring));
        });

        auto*file=menuBar()->addMenu("File");auto*save=file->addAction("Save frame as PNG");
        connect(save,&QAction::triggered,canvas,&Canvas::saveImage);
        auto*quit=file->addAction("Quit");quit->setShortcut(QKeySequence::Quit);connect(quit,&QAction::triggered,this,&QWidget::close);
        auto*more=new QAction(this);more->setShortcut(QKeySequence(Qt::Key_I));addAction(more);
        connect(more,&QAction::triggered,this,[this]{setIterations(canvas->settings.iterations>1000000000?2000000000:canvas->settings.iterations*2);});
        auto*faster=new QAction(this);faster->setShortcut(QKeySequence(Qt::Key_Up));addAction(faster);
        auto*slower=new QAction(this);slower->setShortcut(QKeySequence(Qt::Key_Down));addAction(slower);
        connect(faster,&QAction::triggered,canvas,[this]{canvas->adjustZoomSpeed(true);});
        connect(slower,&QAction::triggered,canvas,[this]{canvas->adjustZoomSpeed(false);});
        auto*stop=new QAction(this);stop->setShortcut(QKeySequence(Qt::Key_Escape));addAction(stop);connect(stop,&QAction::triggered,canvas,&Canvas::stopZoom);
        canvas->onStatus=[this](const QString&s){statusBar()->showMessage(s);};
        statusBar()->showMessage("Calculating; move the pointer and hold left to zoom.");
        resize(1100,800);
    }

    void setIterations(uint32_t iterations) {
        iterations=std::clamp<uint32_t>(iterations,1,2000000000u);
        if(iterations_) {
            iterations_->setValue(static_cast<int>(iterations));
        } else {
            canvas->settings.iterations=iterations;
            canvas->submit(false,true);
            refreshMobileChrome();
        }
    }
    bool mobileUi() const noexcept { return mobile_; }
    /// Verifies that phone controls cannot bubble unhandled touches into Canvas.
    bool mobileInputHierarchyValid() const noexcept {
        return !mobile_ || (mobileDock_ && mobileBadge_ &&
                            mobileDock_->parentWidget()!=canvas &&
                            mobileBadge_->parentWidget()!=canvas);
    }
};
}
/// Starts the Qt desktop application and optional smoke test.
int main(int argc,char**argv) {
    QApplication app(argc,argv);QApplication::setApplicationName("XaoS 30");
#ifdef Q_OS_ANDROID
    const bool mobile=true;
#else
    const bool mobile=app.arguments().contains("--mobile-ui");
#endif
    Window window(mobile);
    const bool smoke=app.arguments().contains("--smoke-test");
    if(smoke) {
        window.resize(mobile?390:520,mobile?760:360);
        window.setIterations(64);
        window.canvas->settings.reconstruction=Reconstruction::Bicubic;
        window.canvas->setThreads(2);
    }
    if(mobile && !smoke) window.showFullScreen(); else window.show();
    if(smoke) {
        struct SmokeState {
            int zoomTicks=0,publishedAtStart=0,finishChecks=0;
            bool publishedDuringMotion=false;
        };
        auto state=std::make_shared<SmokeState>();
        auto*continuous=new QTimer(&window);
        continuous->setInterval(8);
        QObject::connect(continuous,&QTimer::timeout,&window,[&window,state,continuous] {
            state->publishedDuringMotion|=
                window.canvas->publishedFrames>state->publishedAtStart;
            window.canvas->view.zoom(.37,.61,.997,
                std::max(1,window.canvas->width()),std::max(1,window.canvas->height()));
            window.canvas->submit(true);
            if(++state->zoomTicks>=80) {
                state->publishedDuringMotion|=
                    window.canvas->publishedFrames>state->publishedAtStart;
                continuous->stop();
            }
        });
        QTimer::singleShot(150,&window,[&window,state,continuous] {
            state->publishedAtStart=window.canvas->publishedFrames;
            window.canvas->view.zoom(.37,.61,.997,
                std::max(1,window.canvas->width()),std::max(1,window.canvas->height()));
            window.canvas->submit(true);
            state->zoomTicks=1;
            continuous->start();
        });
        QTimer::singleShot(800,&window,[&window]{
            window.canvas->view.rotate(.5,.5,.18,
                std::max(1,window.canvas->width()),std::max(1,window.canvas->height()));
            window.canvas->submit(true);
        });
        QTimer::singleShot(1000,&window,[&window]{window.setIterations(128);});
        QTimer::singleShot(1400,&window,[&window]{window.canvas->settings.minimumPrecision=128;window.canvas->submit(false,true);});
        QTimer::singleShot(1900,&window,[&window]{window.canvas->settings.saveState=false;window.canvas->submit();});
        QTimer::singleShot(2400,&window,[&window]{window.canvas->setAutopilot(true);});
        QTimer::singleShot(2700,&window,[&window]{window.canvas->setPaletteCycling(1);});
        QTimer::singleShot(3200,&window,[&window]{window.canvas->setPaletteCycling(0);});
        QTimer::singleShot(3300,&window,[&window]{window.canvas->setAutopilot(false);});
        // Sanitized builds can be several times slower in presentation, especially
        // now that palette-independent iteration samples are colored on presentation.
        // Exercise the complete scripted scenario first, then give the same assertions
        // a bounded grace period rather than turning machine speed into a test result.
        auto*finish=new QTimer(&window);
        finish->setInterval(100);
        QObject::connect(finish,&QTimer::timeout,&window,[&window,&app,state,finish] {
            const bool ok=window.canvas->completedFrames && state->publishedDuringMotion &&
                          window.mobileInputHierarchyValid();
            if(ok || ++state->finishChecks>=75) {
                finish->stop();
                app.exit(ok?0:2);
            }
        });
        QTimer::singleShot(4500,&window,[finish]{finish->start();});
    }
    return app.exec(); // Window destruction joins all render workers before QApplication dies.
}
