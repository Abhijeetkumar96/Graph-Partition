#ifndef _MYTIMER_
#define _MYTIMER_

#include<iostream>
#include<chrono>
#include<string>

using namespace std;

class mytimer {
	decltype(chrono::high_resolution_clock::now()) tstart,tend;
public:
	mytimer() {
		tstart = chrono::high_resolution_clock::now();
	}

	void timetaken() {
		tend = chrono::high_resolution_clock::now();
		auto dur = chrono::duration_cast<chrono::nanoseconds>(tend - tstart);
		cout << "\n\tTime taken ------ " <<
			dur.count() / ((long long)1000 * 1000 * 1000 * 60) << " min  " <<
			(dur.count() / ((long long)1000 * 1000 * 1000)) % 60 << " sec  " <<
			(dur.count() / (1000 * 1000)) % 1000 << " ms  " <<
			(dur.count() / 1000) % 1000 << " us  " <<
			dur.count() % 1000 << " ns\n";
	}

	void timetaken(string s) {
		tend = chrono::high_resolution_clock::now();
		auto dur = chrono::duration_cast<chrono::nanoseconds>(tend - tstart);
		cout << "\n\tTime taken for " << s <<  " is ------ " <<
			dur.count() / ((long long)1000 * 1000 * 1000 * 60) << " min  " <<
			(dur.count() / ((long long)1000 * 1000 * 1000)) % 60 << " sec  " <<
			(dur.count() / (1000 * 1000)) % 1000 << " ms  " <<
			(dur.count() / 1000) % 1000 << " us  " <<
			dur.count() % 1000 << " ns\n";
	}

	void timetaken_ns(string s) {
		tend = chrono::high_resolution_clock::now();
		auto dur = chrono::duration_cast<chrono::nanoseconds>(tend - tstart);
		cout << "\n\tTime taken for " << s <<  " is ------ " << dur.count() << " ns => " <<
			dur.count() / ((long long)1000 * 1000 * 1000 * 60) << " min  " <<
			(dur.count() / ((long long)1000 * 1000 * 1000)) % 60 << " sec  " <<
			(dur.count() / (1000 * 1000)) % 1000 << " ms  " <<
			(dur.count() / 1000) % 1000 << " us  " <<
			dur.count() % 1000 << " ns\n";
	}

	void timetaken_reset(string s) {
		tend = chrono::high_resolution_clock::now();
		auto dur = chrono::duration_cast<chrono::nanoseconds>(tend - tstart);
		cout << "\n\tTime taken for " << s <<  " is ------ " <<
			dur.count() / ((long long)1000 * 1000 * 1000 * 60) << " min  " <<
			(dur.count() / ((long long)1000 * 1000 * 1000)) % 60 << " sec  " <<
			(dur.count() / (1000 * 1000)) % 1000 << " ms  " <<
			(dur.count() / 1000) % 1000 << " us  " <<
			dur.count() % 1000 << " ns\n";
		tstart = chrono::high_resolution_clock::now();
	}
	
	void reset() {
		tstart = chrono::high_resolution_clock::now();
	}
};

#endif
