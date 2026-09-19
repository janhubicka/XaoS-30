// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/autopilot.hpp"
#include "xaos/formulae.hpp"
#include "xaos/renderer.hpp"
#include "qt_executor.hpp"
#include <QApplication>
#include <QAction>
#include <QMenu>
#include <QKeySequence>
#include <QCheckBox>
#include <QColor>
#include <QClipboard>
#include <QComboBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QElapsedTimer>
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
#include <QTransform>
#include <QWheelEvent>
#include <algorithm>
#include <array>
#include <cmath>
#include <charconv>
#include <condition_variable>
#include <functional>
#include <optional>
#include <numbers>

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
    struct PresentationJob { std::shared_ptr<const FrameBase> frame; uint64_t serial,epoch; };
    std::mutex mutex_;
    std::condition_variable_any wake_;
    std::optional<Job> pending_;
    std::shared_ptr<Cancellation> active_;
    std::jthread coordinator_;
    std::mutex presentationMutex_;
    std::condition_variable_any presentationWake_;
    std::optional<PresentationJob> presentationPending_;
    std::shared_ptr<Cancellation> presentationActive_;
    std::jthread presenter_;
    uint64_t serial_=0,shown_=0,epoch_=1;
    QImage image_,fallback_;
    View imageView_,fallbackView_;
    QTimer motion_,idle_,autopilotTimer_;
    QElapsedTimer motionClock_,autopilotClock_;
    Autopilot autopilotEngine_;
    std::shared_ptr<const DisplayFrame> latestDisplay_;
    Statistics latestDisplayStats_;
    double autopilotStep_=0,zoomSpeedScale_=1.0;
    bool autopilotEnabled_=false;
    QPointF pointer_{.5,.5},lastDrag_;
    int direction_=0;
    bool dragging_=false;
    bool mobileUi_=false;
    bool nativeGestureActive_=false,gestureChanged_=false;
    size_t threads_=std::max<size_t>(1,defaultWorkerCount()-presentationWorkerCount());
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
        }
        {
            std::lock_guard lock(presentationMutex_);
            // Geometry-only motion intentionally does not invalidate an older
            // completed grid: drawView() can transform that source image into the
            // current viewport, exactly as classic XaoS keeps showing/refining
            // frames while the zoom target moves. Keep only the newest pending grid.
            presentationPending_=PresentationJob{std::move(frame),serial,epoch};
        }
        presentationWake_.notify_one();
    }

    /// Reconstructs display frames independently from the compute coordinator.
    void presentationLoop(std::stop_token shutdown) {
        ThreadExecutor executor(presentationWorkerCount());
        std::shared_ptr<const DisplayFrame> previous;
        while(!shutdown.stop_requested()) {
            PresentationJob job;
            std::shared_ptr<Cancellation> token;
            {
                std::unique_lock lock(presentationMutex_);
                if(!presentationWake_.wait(lock,shutdown,[&]{return presentationPending_.has_value();}))
                    return;
                job=std::move(*presentationPending_);
                presentationPending_.reset();
                token=std::make_shared<Cancellation>();
                presentationActive_=token;
            }
            try {
                auto display=presentFrame(*job.frame,executor,*token,previous.get());
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
    void coordinator(std::stop_token shutdown) {
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
        while(!shutdown.stop_requested()) {
            Job job;
            std::shared_ptr<Cancellation> token;
            {
                std::unique_lock lock(mutex_);
                if(!wake_.wait(lock,shutdown,[&]{return pending_.has_value();})) return;
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
                if(!frame->stats.complete && !pending_ && !shutdown.stop_requested())
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
    /// Handles native trackpad gestures and touchscreen pinch/rotation gestures.
    bool event(QEvent*event) override {
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
    /// Submits a new render request after the canvas size changes.
    void resizeEvent(QResizeEvent*e) override { QWidget::resizeEvent(e); submit(false,true); }
    /// Starts zooming or panning in response to a mouse press.
    void mousePressEvent(QMouseEvent*e) override {
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
        if(autopilotEnabled_) {e->accept();return;}
        pointer_=e->position();
        if(dragging_) {
            auto d=pointer_-lastDrag_;lastDrag_=pointer_;
            try {view.pan(d.x(),d.y(),std::max(1,width()));submit(true);} catch(const std::exception&ex){if(onStatus)onStatus(ex.what());}
        }
    }
    /// Double-tap/double-click zooms into the touched point in the mobile UI.
    void mouseDoubleClickEvent(QMouseEvent*e) override {
        if(!mobileUi_ || autopilotEnabled_ || e->button()!=Qt::LeftButton) {
            QWidget::mouseDoubleClickEvent(e);return;
        }
        pointer_=e->position();
        try {
            view.zoom(pointer_.x()/std::max(1,width()),pointer_.y()/std::max(1,height()),
                      .5,std::max(1,width()),std::max(1,height()));
            submit(true);
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
    /// Constructs a Canvas instance.
    explicit Canvas(QWidget*parent=nullptr):QWidget(parent) {
        setMouseTracking(true);setFocusPolicy(Qt::StrongFocus);
        setAttribute(Qt::WA_AcceptTouchEvents,true);
        grabGesture(Qt::PinchGesture);
        motion_.setInterval(16);idle_.setSingleShot(true);idle_.setInterval(180);
        autopilotTimer_.setInterval(40); // original XaoS autopilot timer: 25 Hz
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
        connect(&autopilotTimer_,&QTimer::timeout,this,[this]{autopilotTick();});
        presenter_=std::jthread([this](std::stop_token s){presentationLoop(s);});
        coordinator_=std::jthread([this](std::stop_token s){coordinator(s);});
    }
    /// Releases resources owned by the Canvas instance.
    ~Canvas() override {
        motion_.stop();idle_.stop();autopilotTimer_.stop();
        coordinator_.request_stop();
        presenter_.request_stop();
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
        direction_=0;motion_.stop();dragging_=false;
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
    void setMobileUi(bool enabled) { mobileUi_=enabled;update(); }
    /// Returns whether phone-oriented interaction is enabled.
    bool mobileUi() const noexcept { return mobileUi_; }
    /// Returns the number of compute workers, excluding presentation workers.
    size_t workerCount() const noexcept { return threads_; }
    /// Changes the worker count and requests a new render.
    void setThreads(size_t n) {threads_=n;submit();}
    /// Restores the default fractal view and requests a render.
    void reset() {restoreFormulaDefault();submit();}
    /// Stops continuous zooming and requests refinement of the current view.
    void stopZoom() {direction_=0;motion_.stop();setAutopilot(false);submit();}
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
        mobileQuality_->setText(qualityName()+"\nLOOK");
        mobileExplore_->setText(canvas->autopilotEnabled()?"STOP\nEXPLORE":"EXPLORE\nAUTO");
        mobileExplore_->setChecked(canvas->autopilotEnabled());
        mobileBadge_->adjustSize();
        layoutMobileChrome();
    }
    void layoutMobileChrome() {
        if(!mobile_ || !mobileDock_ || !mobileBadge_) return;
        const int w=canvas->width(),h=canvas->height();
        const int margin=std::clamp(w/28,12,22);
        const int dockHeight=std::clamp(h/10,68,88);
        mobileDock_->setGeometry(margin,h-dockHeight-margin,std::max(120,w-2*margin),dockHeight);
        mobileBadge_->adjustSize();
        mobileBadge_->move(margin,margin);
        mobileDock_->raise();mobileBadge_->raise();
    }
    QToolButton* mobileButton(const QString&text,QWidget*parent) {
        auto*b=new QToolButton(parent);
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

        mobileBadge_=new QLabel(canvas);
        mobileBadge_->setAttribute(Qt::WA_StyledBackground,true);
        mobileBadge_->setStyleSheet(
            "QLabel{color:white;background:rgba(8,10,16,190);"
            "border:1px solid rgba(255,255,255,36);border-radius:16px;"
            "padding:7px 12px;font-size:14px;font-weight:650;}");

        mobileDock_=new QFrame(canvas);
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
        mobileQuality_=mobileButton("CRISP\nLOOK",mobileDock_);
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

        mobileMoreMenu_=new QMenu(more);
        auto*coordinates=mobileMoreMenu_->addAction("Coordinates & precision");
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
        connect(mobileQuality_,&QToolButton::clicked,this,[this]{
            const int next=(static_cast<int>(canvas->settings.reconstruction)+1)%3;
            canvas->settings.reconstruction=static_cast<Reconstruction>(next);
            canvas->submit(false,true);refreshMobileChrome();
        });
        connect(reset,&QToolButton::clicked,canvas,[this]{canvas->reset();refreshMobileChrome();});
        connect(coordinates,&QAction::triggered,canvas,&Canvas::coordinates);
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
                "One finger drags the plane.\n"
                "Pinch with two fingers to zoom.\n"
                "Twist two fingers to rotate.\n"
                "Double-tap to dive in.\n\n"
                "Explore lets XaoS choose the next interesting boundary automatically.");
        });
        canvas->onAutopilotChanged=[this](bool){refreshMobileChrome();};
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
            int zoomTicks=0,publishedAtStart=0;
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
        QTimer::singleShot(3300,&window,[&window]{window.canvas->setAutopilot(false);});
        QTimer::singleShot(4500,&window,[&window,&app,state]{
            const bool ok=window.canvas->completedFrames && state->publishedDuringMotion;
            app.exit(ok?0:2);
        });
    }
    return app.exec(); // Window destruction joins all render workers before QApplication dies.
}
