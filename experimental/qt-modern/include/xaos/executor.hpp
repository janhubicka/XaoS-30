// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
namespace xaos {
size_t defaultWorkerCount() noexcept;
struct Cancellation {
    std::atomic<bool> cancelled{false};
    const Cancellation* parent=nullptr; // non-owning; parent outlives the synchronous render
    std::chrono::steady_clock::time_point deadline=std::chrono::steady_clock::time_point::max();
    bool requested(bool timeBudget=true) const noexcept {
        return cancelled.load(std::memory_order_relaxed) || (parent && parent->requested(timeBudget)) ||
            (timeBudget && deadline!=std::chrono::steady_clock::time_point::max() && std::chrono::steady_clock::now()>=deadline);
    }
};
class Executor {
public:
    virtual ~Executor()=default;
    virtual size_t concurrency() const noexcept=0;
    // Synchronous barrier: no callback may survive return, including exceptions.
    // Called only by one coordinator; do not call recursively from its workers.
    virtual void run(const std::function<void(size_t)>& work)=0;
};
class ThreadExecutor final:public Executor {
    std::mutex mutex_;
    std::condition_variable_any wake_;
    std::condition_variable finished_;
    std::vector<std::jthread> threads_;
    std::function<void(size_t)> work_;
    std::exception_ptr error_;
    size_t generation_=0,remaining_=0;
public:
    explicit ThreadExecutor(size_t count);
    ~ThreadExecutor() override;
    size_t concurrency() const noexcept override { return threads_.size(); }
    void run(const std::function<void(size_t)>& work) override;
};
}
