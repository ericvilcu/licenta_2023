#pragma once
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#ifdef __NVCC__
static_assert(false, "DO NOT INCLUDE THIS FILE FROM .cu FILES! IT UTILIZES 'typedef' WHICH CURRENTLY BREAKS NVCC COMPILED FILES!");
#endif
template <typename T>
class BlockingQueue {
private:
	std::queue<T> queue;
	std::mutex m;
	std::condition_variable on_put;
	std::condition_variable on_take;
	int max_size;
	bool valid = true;
	T err;
public:
	BlockingQueue(int max_size, T err) 
		:max_size{ max_size }, err{ err }, m{}, on_put{}, on_take{}
		{}
	//They need to be moved. This should usually use shared pointers, which can cause errors on threads. Therefore we should remove any trace of them right before being done.
	int push(T&& item) {
		std::unique_lock<std::mutex> local_lock{ m };
		while (queue.size() >= max_size && valid) {
			on_take.wait(local_lock);
		}
		if (!valid) return -1;
		queue.push(std::move(item));
		on_put.notify_one();
		return 0;
	}
	T pop() {
		std::unique_lock<std::mutex> local_lock{ m };
		while (queue.size() <= 0 && valid) {
			on_put.wait(local_lock);
		}
		if (!valid) return err;
		T ret = queue.front();
		queue.pop();
		on_take.notify_one();
		return ret;
	}
	void invalidate() {
		std::unique_lock<std::mutex> local_lock{ m };
		valid = false;
		on_put.notify_all(); on_take.notify_all();
	}
	template <typename COMM>
	class producer {
	private:
		typedef std::function<std::pair<COMM, T>(COMM)> fun_type;
		fun_type f;
		COMM current;
		std::thread me;
		BlockingQueue& queue;
	public:
		producer(COMM start, fun_type f, BlockingQueue& queue) :f{ f }, current{ start }, me{ run, this }, queue{ queue } {};
		static void run(producer* ths) {
			try {
				while (true) {
					auto rez = ths->f(ths->current);
					ths->current = rez.first;
					if (ths->queue.push(std::move(rez.second)) != 0)return;
				}
			} catch (...) {
				ths->queue.invalidate();
			}
		}
		~producer() {
			queue.invalidate();
			me.join();
		}
	};
};
