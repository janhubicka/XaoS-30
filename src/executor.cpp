// SPDX-License-Identifier: GPL-2.0-or-later
#include "xaos/executor.hpp"
#include <stdexcept>
#include <algorithm>
#ifdef __linux__
#include <sched.h>
#endif
namespace xaos {
/// Returns a conservative default number of worker threads.
size_t defaultWorkerCount() noexcept {
    size_t n=std::max(1u,std::thread::hardware_concurrency());
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if(sched_getaffinity(0,sizeof(set),&set)==0) n=std::max(1,CPU_COUNT(&set));
#endif
    return std::min<size_t>(1024,n);
}

/// Constructs a ThreadExecutor instance.
ThreadExecutor::ThreadExecutor(size_t count) {
    if(count<1||count>1024) throw std::invalid_argument("worker count must be between 1 and 1024");
    threads_.reserve(count);
    for(size_t id=0;id<count;++id) threads_.emplace_back([this,id](std::stop_token stop) {
        size_t seen=0;
        while(true) {
            std::unique_lock lock(mutex_);
            if(!wake_.wait(lock,stop,[&]{return generation_!=seen;})) return;
            seen=generation_;
            auto work=work_;
            lock.unlock();
            try { work(id); }
            catch(...) { std::lock_guard guard(mutex_); if(!error_) error_=std::current_exception(); }
            lock.lock();
            if(--remaining_==0) finished_.notify_one();
        }
    });
}
/// Releases resources owned by the ThreadExecutor instance.
ThreadExecutor::~ThreadExecutor() {
    for(auto&t:threads_) t.request_stop();
    wake_.notify_all();
    for(auto&t:threads_) if(t.joinable()) t.join();
}
/// Performs the run operation.
void ThreadExecutor::run(const std::function<void(size_t)>& work) {
    std::unique_lock lock(mutex_);
    if(remaining_) throw std::logic_error("concurrent/nested Executor::run");
    error_=nullptr; work_=work; remaining_=threads_.size(); ++generation_;
    wake_.notify_all();
    finished_.wait(lock,[&]{return remaining_==0;});
    work_={};
    if(error_) std::rethrow_exception(error_);
}
}
