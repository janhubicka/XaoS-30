// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include "xaos/executor.hpp"
#include <QThreadPool>
#include <QRunnable>
#include <memory>
#include <stdexcept>

// Qt owns the persistent pixel workers. The render coordinator is a different
// thread, so a pool worker never blocks waiting for work in the same pool.
class QtExecutor final:public xaos::Executor {
    QThreadPool pool_;
    size_t count_;
public:
    /// Constructs a QtExecutor instance.
    explicit QtExecutor(size_t count):count_(count) {
        if(count<1 || count>1024) throw std::invalid_argument("invalid Qt worker count");
        pool_.setMaxThreadCount(static_cast<int>(count));
        pool_.setExpiryTimeout(-1);
    }
    /// Releases resources owned by the QtExecutor instance.
    ~QtExecutor() override { pool_.waitForDone(); }
    /// Returns the number of workers available to the executor.
    size_t concurrency() const noexcept override { return count_; }
    /// Executes scheduled work using the implementation-specific worker machinery.
    void run(const std::function<void(size_t)>&work) override {
        struct Batch {
            std::mutex mutex;
            std::condition_variable done;
            size_t remaining=0;
            std::exception_ptr error;
            std::function<void(size_t)> work;
        };
        auto b=std::make_shared<Batch>(); b->remaining=count_; b->work=work;
        std::vector<std::unique_ptr<QRunnable>> tasks;
        tasks.reserve(count_);
        // Allocate everything before submission: allocation failure leaves no
        // queued task referring to a callback whose captures have gone away.
        for(size_t id=0;id<count_;++id) tasks.emplace_back(QRunnable::create([b,id] {
            try { b->work(id); }
            catch(...) { std::lock_guard lock(b->mutex); if(!b->error) b->error=std::current_exception(); }
            std::lock_guard lock(b->mutex);
            if(--b->remaining==0) b->done.notify_one();
        }));
        for(auto&task:tasks) pool_.start(task.release());
        std::unique_lock lock(b->mutex);
        b->done.wait(lock,[&]{return b->remaining==0;});
        if(b->error) std::rethrow_exception(b->error);
    }
};
