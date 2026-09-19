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
#include <QImage>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMenuBar>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QSpinBox>
#include <QStatusBar>
#include <QTimer>
#include <QToolBar>
#include <QWheelEvent>
#include <algorithm>
#include <array>
#include <cmath>
#include <charconv>
#include <condition_variable>
#include <functional>
#include <optional>

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
    size_t threads_=std::max<size_t>(1,defaultWorkerCount()-presentationWorkerCount());
    /// Maps a target selected in a displayed source frame into the current viewport.
    QPointF mapDisplayFocus(const DisplayFrame&frame,double u,double v) const {
        if(width()<1 || height()<1) return QPointF(.5,.5);
        const auto&source=frame.request.view;
        const double sourceAspect=static_cast<double>(frame.request.height)/frame.request.width;
        Big real=add(source.re,scale(source.span,u-.5));
        Big imag=add(source.im,scale(source.span,(.5-v)*sourceAspect));
        const double currentU=.5+div(sub(real,view.re),view.span).toDouble();
        const double currentV=.5-div(sub(imag,view.im),view.span).toDouble()*
                                   static_cast<double>(width())/height();
        return QPointF(std::clamp(currentU,0.0,1.0)*width(),
                       std::clamp(currentV,0.0,1.0)*height());
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
    /// Draws a cached image transformed into the current viewport, including zoom-out edge extension.
    void drawView(QPainter&p,const QImage&image,const View&source) {
        if(image.isNull() || width()<1||height()<1) return;
        const Big oldLeft=sub(source.re,scale(source.span,.5));
        const Big newLeft=sub(view.re,scale(view.span,.5));
        const Big oldTop=add(source.im,scale(source.span,.5*image.height()/image.width()));
        const Big newTop=add(view.im,scale(view.span,.5*height()/width()));
        // Convert only *relative screen coordinates* to double, after subtraction
        // and scaling at arbitrary precision. Never subtract two rounded doubles.
        const double x=div(sub(oldLeft,newLeft),view.span).toDouble()*width();
        const double y=div(sub(newTop,oldTop),view.span).toDouble()*width();
        const double w=div(source.span,view.span).toDouble()*width();
        const double h=w*image.height()/image.width();
        if(std::isfinite(x)&&std::isfinite(y)&&std::isfinite(w)&&std::isfinite(h) && w>0 && w<1.e9) {
            // During zoom-out the transformed previous frame is smaller than the
            // widget. Classic XaoS immediately fills the newly exposed bands from
            // the nearest boundary row/column instead of flashing black. Do the
            // same while the next DP frame is still being computed.
            const double cw=width(),ch=height();
            if(x>0) {
                p.drawImage(QRectF(0,y,x,h),image,QRectF(0,0,1,image.height()));
                if(y>0) p.fillRect(QRectF(0,0,x,y),QColor::fromRgba(image.pixel(0,0)));
                if(y+h<ch) p.fillRect(QRectF(0,y+h,x,ch-(y+h)),QColor::fromRgba(image.pixel(0,image.height()-1)));
            }
            if(x+w<cw) {
                p.drawImage(QRectF(x+w,y,cw-(x+w),h),image,
                            QRectF(image.width()-1,0,1,image.height()));
                if(y>0) p.fillRect(QRectF(x+w,0,cw-(x+w),y),
                                  QColor::fromRgba(image.pixel(image.width()-1,0)));
                if(y+h<ch) p.fillRect(QRectF(x+w,y+h,cw-(x+w),ch-(y+h)),
                                     QColor::fromRgba(image.pixel(image.width()-1,image.height()-1)));
            }
            if(y>0)
                p.drawImage(QRectF(x,0,w,y),image,QRectF(0,0,image.width(),1));
            if(y+h<ch)
                p.drawImage(QRectF(x,y+h,w,ch-(y+h)),image,
                            QRectF(0,image.height()-1,image.width(),1));
            p.drawImage(QRectF(x,y,w,h),image);
        }
    }
protected:
    /// Paints the current and fallback fractal images plus the interaction hint.
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        // QPainter backends differ in their default image-scaling behavior. XaoS
        // is a sample-reuse zoomer, not a bilinear image zoomer: keep moved rows
        // and columns crisp while new resolution is being computed.
        p.setRenderHint(QPainter::SmoothPixmapTransform,false);
        p.fillRect(rect(),Qt::black);
        drawView(p,fallback_,fallbackView_);drawView(p,image_,imageView_);
        p.setPen(Qt::white);
        p.drawText(12,22,"Hold left/right: zoom   |   Middle drag: pan   |   Wheel: zoom   |   A: autopilot");
    }
    /// Submits a new render request after the canvas size changes.
    void resizeEvent(QResizeEvent*e) override { QWidget::resizeEvent(e); submit(false,true); }
    /// Starts zooming or panning in response to a mouse press.
    void mousePressEvent(QMouseEvent*e) override {
        if(autopilotEnabled_) {e->accept();return;}
        pointer_=e->position();
        if(e->button()==Qt::MiddleButton) {dragging_=true;lastDrag_=pointer_;}
        else if(e->button()==Qt::LeftButton || e->button()==Qt::RightButton) {
            direction_=e->button()==Qt::LeftButton?1:-1;motionClock_.restart();motion_.start();
        }
    }
    /// Stops the active mouse interaction and schedules idle refinement.
    void mouseReleaseEvent(QMouseEvent*e) override {
        if(autopilotEnabled_) {e->accept();return;}
        if(e->button()==Qt::MiddleButton) dragging_=false;
        if(e->button()==Qt::LeftButton || e->button()==Qt::RightButton) {direction_=0;motion_.stop();idle_.start();}
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
        QLineEdit precision(QString::number(static_cast<qulonglong>(settings.minimumPrecision)));
        QLineEdit jr,ji;
        jr.setMaxLength(std::numeric_limits<int>::max());ji.setMaxLength(std::numeric_limits<int>::max());
        jr.setText(QString::fromStdString(settings.juliaRe.str()));ji.setText(QString::fromStdString(settings.juliaIm.str()));
        form.addRow("Center, real",&re);form.addRow("Center, imaginary",&im);form.addRow("Horizontal span",&span);
        form.addRow("Minimum bits (0 = adaptive)",&precision);form.addRow("Julia c, real",&jr);form.addRow("Julia c, imaginary",&ji);
        QLabel note("Precision grows with zoom depth. More bits invalidate old orbits.\nMemory and CPU time remain finite; direct GMP is not perturbation rendering.");
        form.addRow(&note);
        QDialogButtonBox buttons(QDialogButtonBox::Ok|QDialogButtonBox::Cancel);
        form.addRow(&buttons);connect(&buttons,&QDialogButtonBox::accepted,&d,&QDialog::accept);connect(&buttons,&QDialogButtonBox::rejected,&d,&QDialog::reject);
        d.resize(650,d.sizeHint().height());
        if(d.exec()==QDialog::Accepted) {
            try {
                View next=View::parse(re.text().trimmed().toStdString(),im.text().trimmed().toStdString(),span.text().trimmed().toStdString(),std::max(1,width()),16,settings.memoryBudget);
                auto bits=readInteger<mp_bitcnt_t>(precision.text());
                const auto r=jr.text().trimmed().toStdString(),i=ji.text().trimmed().toStdString();
                Big real=Big::parse(r,View::textBits(r,i,"")),imag=Big::parse(i,View::textBits(r,i,""));
                view=std::move(next);settings.minimumPrecision=bits;settings.juliaRe=std::move(real);settings.juliaIm=std::move(imag);submit(false,true);
            }catch(const std::exception&e){QMessageBox::warning(this,"Invalid view",e.what());}
        }
    }
};
class Window final:public QMainWindow {
public:
    Canvas*canvas;
    QSpinBox*iterations;
    /// Constructs a Window instance.
    Window() {
        canvas=new Canvas(this);setCentralWidget(canvas);
        setWindowTitle("XaoS Modern — reusable orbits / arbitrary precision");
        auto*bar=addToolBar("Rendering");bar->setMovable(false);
        auto*formula=new QComboBox(bar);
        for(const auto&info:formulaInfos())
            formula->addItem(QString::fromLatin1(info.name),static_cast<int>(info.formula));
        bar->addWidget(formula);
        bar->addWidget(new QLabel("  Iterations ",bar));iterations=new QSpinBox(bar);iterations->setRange(1,2000000000);iterations->setValue(512);bar->addWidget(iterations);
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
        connect(iterations,qOverload<int>(&QSpinBox::valueChanged),this,[this](int n){canvas->settings.iterations=static_cast<uint32_t>(n);canvas->submit(false,true);});
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
        connect(more,&QAction::triggered,this,[this]{iterations->setValue(iterations->value()>1000000000?2000000000:iterations->value()*2);});
        auto*faster=new QAction(this);faster->setShortcut(QKeySequence(Qt::Key_Up));addAction(faster);
        auto*slower=new QAction(this);slower->setShortcut(QKeySequence(Qt::Key_Down));addAction(slower);
        connect(faster,&QAction::triggered,canvas,[this]{canvas->adjustZoomSpeed(true);});
        connect(slower,&QAction::triggered,canvas,[this]{canvas->adjustZoomSpeed(false);});
        auto*stop=new QAction(this);stop->setShortcut(QKeySequence(Qt::Key_Escape));addAction(stop);connect(stop,&QAction::triggered,canvas,&Canvas::stopZoom);
        canvas->onStatus=[this](const QString&s){statusBar()->showMessage(s);};
        statusBar()->showMessage("Calculating; move the pointer and hold left to zoom.");
        resize(1100,800);
    }
};
}
/// Starts the Qt desktop application and optional smoke test.
int main(int argc,char**argv) {
    QApplication app(argc,argv);QApplication::setApplicationName("XaoS Modern");
    Window window;
    const bool smoke=app.arguments().contains("--smoke-test");
    if(smoke) {
        window.resize(520,360);
        window.iterations->setValue(64);
        window.canvas->settings.reconstruction=Reconstruction::Bicubic;
        window.canvas->setThreads(2);
    }
    window.show();
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
        QTimer::singleShot(1000,&window,[&window]{window.iterations->setValue(128);});
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
