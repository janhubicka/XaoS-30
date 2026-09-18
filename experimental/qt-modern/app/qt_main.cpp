// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/renderer.hpp"
#include "qt_executor.hpp"
#include <QApplication>
#include <QAction>
#include <QMenu>
#include <QKeySequence>
#include <QCheckBox>
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
#include <charconv>
#include <condition_variable>
#include <functional>
#include <optional>

using namespace xaos;
namespace {
template<class T> T readInteger(const QString&text) {
    const auto s=text.trimmed().toStdString(); T n{};
    auto [p,e]=std::from_chars(s.data(),s.data()+s.size(),n);
    if(e!=std::errc{} || p!=s.data()+s.size()) throw std::invalid_argument("invalid integer");
    return n;
}
QImage makeImage(const FrameBase&f) {
    QImage image(f.request.width,f.request.height,QImage::Format_ARGB32_Premultiplied);
    if(image.isNull()) throw std::bad_alloc();
    // This QImage has a single owner until publication. No worker paints it.
    for(int row=0;row<f.request.height;++row) {
        auto*data=reinterpret_cast<QRgb*>(image.scanLine(row));
        const int y=f.request.height-1-row;
        for(int x=0;x<f.request.width;++x) {
            Count c=f.at(x,y);
            data[x]=c.known(f.request.settings.iterations)?pixelColor(c,f.request.settings.iterations):0;
        }
    }
    return image;
}
class Canvas final:public QWidget {
    struct Job { Request request; size_t threads; uint64_t serial; };
    std::mutex mutex_;
    std::condition_variable_any wake_;
    std::optional<Job> pending_;
    std::shared_ptr<Cancellation> active_;
    std::jthread coordinator_;
    uint64_t serial_=0,shown_=0;
    QImage image_,fallback_;
    View imageView_,fallbackView_;
    QTimer motion_,idle_;
    QElapsedTimer motionClock_;
    QPointF pointer_{.5,.5},lastDrag_;
    int direction_=0;
    bool dragging_=false;
    size_t threads_=defaultWorkerCount();
    void coordinator(std::stop_token shutdown) {
        Renderer renderer;
        std::unique_ptr<QtExecutor> executor;
        while(!shutdown.stop_requested()) {
            Job job;
            std::shared_ptr<Cancellation> token;
            {
                std::unique_lock lock(mutex_);
                if(!wake_.wait(lock,shutdown,[&]{return pending_.has_value();})) return;
                job=std::move(*pending_);pending_.reset();
                token=std::make_shared<Cancellation>();
                job.request.settings.sliceMilliseconds=job.request.settings.uniform?100:40;
                active_=token;
            }
            try {
                if(!executor || executor->concurrency()!=job.threads) executor=std::make_unique<QtExecutor>(job.threads);
                auto frame=renderer.render(job.request,*executor,*token);
                auto image=makeImage(*frame); const auto stats=frame->stats;
                const auto view=frame->request.view;
                // No QWidget access on this thread. QObject drops queued calls on
                // destruction; our destructor also joins this coordinator first.
                QMetaObject::invokeMethod(this,[this,image=std::move(image),view,stats,id=job.serial] {
                    if(id<shown_) return;
                    shown_=id;
                    fallback_=image_;fallbackView_=imageView_;
                    image_=image;imageView_=view;
                    if(stats.complete) ++completedFrames;
                    if(onStatus) onStatus(QString("%1%2  |  %3 bits  |  %4 ms  |  reused %5  resumed %6  |  %7%8")
                        .arg(QString::fromStdString(stats.backend)).arg(stats.simd?" / AVX2":"")
                        .arg(static_cast<qulonglong>(stats.bits)).arg(stats.milliseconds,0,'f',1)
                        .arg(static_cast<qulonglong>(stats.reused)).arg(static_cast<qulonglong>(stats.resumed))
                        .arg(stats.uniform?"uniform samples":"adaptive preview")
                        .arg(stats.complete?"":" / refining"));
                    update();
                },Qt::QueuedConnection);
                std::lock_guard lock(mutex_);
                if(active_==token) active_.reset();
                // Continue the same view in bounded slices when input has stopped.
                // Count-only kernels finish each orbit before their time-budget yield.
                if(!frame->stats.complete && !pending_ && !shutdown.stop_requested()) pending_=job;
            } catch(const std::exception&e) {
                const QString message=QString::fromUtf8(e.what());
                QMetaObject::invokeMethod(this,[this,message,id=job.serial] {
                    if(id==serial_ && onStatus) onStatus("Render error: "+message);
                },Qt::QueuedConnection);
                std::lock_guard lock(mutex_);if(active_==token) active_.reset();
            }
        }
    }
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
        if(std::isfinite(x)&&std::isfinite(y)&&std::isfinite(w)&&std::isfinite(h) && w>0 && w<1.e9)
            p.drawImage(QRectF(x,y,w,h),image);
    }
protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);p.fillRect(rect(),Qt::black);
        drawView(p,fallback_,fallbackView_);drawView(p,image_,imageView_);
        p.setPen(Qt::white);
        p.drawText(12,22,"Hold left/right: zoom   |   Middle drag: pan   |   Wheel: zoom   |   I: more iterations");
    }
    void resizeEvent(QResizeEvent*e) override { QWidget::resizeEvent(e); submit(false); }
    void mousePressEvent(QMouseEvent*e) override {
        pointer_=e->position();
        if(e->button()==Qt::MiddleButton) {dragging_=true;lastDrag_=pointer_;}
        else if(e->button()==Qt::LeftButton || e->button()==Qt::RightButton) {
            direction_=e->button()==Qt::LeftButton?1:-1;motionClock_.restart();motion_.start();
        }
    }
    void mouseReleaseEvent(QMouseEvent*e) override {
        if(e->button()==Qt::MiddleButton) dragging_=false;
        if(e->button()==Qt::LeftButton || e->button()==Qt::RightButton) {direction_=0;motion_.stop();idle_.start();}
    }
    void mouseMoveEvent(QMouseEvent*e) override {
        pointer_=e->position();
        if(dragging_) {
            auto d=pointer_-lastDrag_;lastDrag_=pointer_;
            try {view.pan(d.x(),d.y(),std::max(1,width()));submit(false);} catch(const std::exception&ex){if(onStatus)onStatus(ex.what());}
        }
    }
    void wheelEvent(QWheelEvent*e) override {
        const double steps=e->angleDelta().y()/120.;
        try {
            view.zoom(e->position().x()/std::max(1,width()),e->position().y()/std::max(1,height()),
                      std::exp(-.2*steps),std::max(1,width()),std::max(1,height()));submit(false);
        }catch(const std::exception&ex){if(onStatus)onStatus(ex.what());}
        e->accept();
    }
public:
    View view;
    Settings settings;
    int completedFrames=0;
    std::function<void(QString)> onStatus;
    explicit Canvas(QWidget*parent=nullptr):QWidget(parent) {
        setMouseTracking(true);setFocusPolicy(Qt::StrongFocus);
        motion_.setInterval(16);idle_.setSingleShot(true);idle_.setInterval(180);
        connect(&motion_,&QTimer::timeout,this,[this] {
            if(!direction_) return;
            const double seconds=std::min<qint64>(motionClock_.restart(),100)/1000.;
            try {
                view.zoom(pointer_.x()/std::max(1,width()),pointer_.y()/std::max(1,height()),
                          std::exp(-direction_*seconds*.75),std::max(1,width()),std::max(1,height()));submit(false);
            }catch(const std::exception&e){motion_.stop();if(onStatus)onStatus(e.what());}
        });
        connect(&idle_,&QTimer::timeout,this,[this]{submit(true);});
        coordinator_=std::jthread([this](std::stop_token s){coordinator(s);});
    }
    ~Canvas() override {
        motion_.stop();idle_.stop();
        coordinator_.request_stop();
        {std::lock_guard lock(mutex_);pending_.reset();if(active_)active_->cancelled.store(true,std::memory_order_relaxed);}
        wake_.notify_all();
        if(coordinator_.joinable()) coordinator_.join();
    }
    void submit(bool uniform=true) {
        if(width()<1||height()<1) return;
        const double dpr=devicePixelRatioF();
        Request request{view,std::max(1,static_cast<int>(std::ceil(width()*dpr))),
                             std::max(1,static_cast<int>(std::ceil(height()*dpr))),settings};
        request.settings.uniform=uniform;
        request.settings.focusX=pointer_.x()/width();request.settings.focusY=pointer_.y()/height();
        {
            std::lock_guard lock(mutex_);
            if(active_) active_->cancelled.store(true,std::memory_order_relaxed);
            pending_=Job{std::move(request),threads_,++serial_};
        }
        wake_.notify_one();update();
        if(!uniform) idle_.start();
    }
    void setThreads(size_t n) {threads_=n;submit();}
    void reset() {view=View{};submit();}
    void stopZoom() {direction_=0;motion_.stop();submit();}
    void saveImage() {
        if(image_.isNull()) return;
        const auto path=QFileDialog::getSaveFileName(this,"Save displayed frame",{},"PNG (*.png)");
        if(!path.isEmpty() && !image_.save(path)) QMessageBox::warning(this,"Save failed","Could not write the image.");
    }
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
                view=std::move(next);settings.minimumPrecision=bits;settings.juliaRe=std::move(real);settings.juliaIm=std::move(imag);submit();
            }catch(const std::exception&e){QMessageBox::warning(this,"Invalid view",e.what());}
        }
    }
};
class Window final:public QMainWindow {
public:
    Canvas*canvas;
    QSpinBox*iterations;
    Window() {
        canvas=new Canvas(this);setCentralWidget(canvas);
        setWindowTitle("XaoS Modern — reusable orbits / arbitrary precision");
        auto*bar=addToolBar("Rendering");bar->setMovable(false);
        auto*formula=new QComboBox(bar);formula->addItems({"Mandelbrot","Julia","Burning ship"});bar->addWidget(formula);
        bar->addWidget(new QLabel("  Iterations ",bar));iterations=new QSpinBox(bar);iterations->setRange(1,2000000000);iterations->setValue(512);bar->addWidget(iterations);
        auto*states=new QCheckBox("Save orbits",bar);states->setChecked(true);bar->addWidget(states);
        bar->addWidget(new QLabel("  Workers ",bar));auto*threads=new QSpinBox(bar);threads->setRange(1,1024);
        threads->setValue(static_cast<int>(defaultWorkerCount()));bar->addWidget(threads);
        auto*coords=bar->addAction("Coordinates / bits");auto*reset=bar->addAction("Reset");
        connect(formula,qOverload<int>(&QComboBox::currentIndexChanged),this,[this](int i){canvas->settings.formula=static_cast<Formula>(i);canvas->submit();});
        connect(iterations,qOverload<int>(&QSpinBox::valueChanged),this,[this](int n){canvas->settings.iterations=static_cast<uint32_t>(n);canvas->submit();});
        connect(states,&QCheckBox::toggled,this,[this](bool b){canvas->settings.saveState=b;canvas->submit();});
        connect(threads,qOverload<int>(&QSpinBox::valueChanged),this,[this](int n){canvas->setThreads(static_cast<size_t>(n));});
        connect(coords,&QAction::triggered,canvas,&Canvas::coordinates);connect(reset,&QAction::triggered,canvas,&Canvas::reset);
        auto*file=menuBar()->addMenu("File");auto*save=file->addAction("Save frame as PNG");
        connect(save,&QAction::triggered,canvas,&Canvas::saveImage);
        auto*quit=file->addAction("Quit");quit->setShortcut(QKeySequence::Quit);connect(quit,&QAction::triggered,this,&QWidget::close);
        auto*more=new QAction(this);more->setShortcut(QKeySequence(Qt::Key_I));addAction(more);
        connect(more,&QAction::triggered,this,[this]{iterations->setValue(iterations->value()>1000000000?2000000000:iterations->value()*2);});
        auto*stop=new QAction(this);stop->setShortcut(QKeySequence(Qt::Key_Escape));addAction(stop);connect(stop,&QAction::triggered,canvas,&Canvas::stopZoom);
        canvas->onStatus=[this](const QString&s){statusBar()->showMessage(s);};
        statusBar()->showMessage("Calculating; move the pointer and hold left to zoom.");
        resize(1100,800);
    }
};
}
int main(int argc,char**argv) {
    QApplication app(argc,argv);QApplication::setApplicationName("XaoS Modern");
    Window window;
    const bool smoke=app.arguments().contains("--smoke-test");
    if(smoke) {window.resize(420,320);window.iterations->setValue(64);window.canvas->setThreads(2);}
    window.show();
    if(smoke) {
        QTimer::singleShot(200,&window,[&]{window.canvas->view.zoom(.4,.6,.97,std::max(1,window.canvas->width()),std::max(1,window.canvas->height()));window.canvas->submit(false);});
        QTimer::singleShot(400,&window,[&]{window.iterations->setValue(128);});
        QTimer::singleShot(600,&window,[&]{window.canvas->settings.minimumPrecision=128;window.canvas->submit();});
        QTimer::singleShot(1200,&window,[&]{window.canvas->settings.saveState=false;window.canvas->submit();});
        QTimer::singleShot(3000,&window,[&]{app.exit(window.canvas->completedFrames?0:2);});
    }
    return app.exec(); // Window destruction joins all render workers before QApplication dies.
}
