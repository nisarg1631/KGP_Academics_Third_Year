#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <pthread.h>
#include <stack>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <sys/resource.h>
using namespace std;

const int MAT_SIZE = 100;
const int MAX_QUEUE_SIZE = 10;
const int RANDOM_ID_RANGE_LO = 1;
const int RANDOM_ID_RANGE_HI = 100000;

static random_device rd;
static mt19937 generator(rd());

int total_jobs;
vector<vector<int>> get_random_vectors(int cnt) {
    vector<int> v(RANDOM_ID_RANGE_HI - RANDOM_ID_RANGE_LO + 1);
    iota(v.begin(), v.end(), 1);
    // random_shuffle(v.begin(), v.end());
    shuffle(v.begin(), v.end(), generator);
    vector<vector<int>> r(cnt);
    int it = 0;
    while(!v.empty()) {
        r[it].push_back(v.back());
        it = (it + 1) % cnt;
        v.pop_back();
    }
    return r;
}

void foo1(int &n) {
    n = 10;
}

void foo2(int &n) {
    foo1(n);
}

const int MATRIX_ELEMENT_LO = -9;
const int MATRIX_ELEMENT_HI = 9;
const int WAIT_TIME_LO = 0;
const int WAIT_TIME_HI = 3000;

int get_random_matrix_entry() {
    static uniform_int_distribution<int> range(MATRIX_ELEMENT_LO, MATRIX_ELEMENT_HI);
    return range(generator);
}

int get_random_wait_time() {
    static uniform_int_distribution<int> range(WAIT_TIME_LO, WAIT_TIME_HI);
    return range(generator);
}

signed main(int argc, char *argv[]) {
    // auto vecs = get_random_vectors(atoi(argv[1]));
    // for(auto i:vecs) {
    //     for(auto j:i) cout << j << ' ';
    //     cout << '\n';
    // }
    // int arr[3001] = {0};
    // for(int i=0; i<300000; i++) {
    //     arr[get_random_wait_time()]++;
    // }
    // for(int i=0; i<3001; i++) cout << i << " : " << arr[i] << endl;
    // cout << accumulate(arr, arr+3001, 0) << endl;

    // for(int i=0; i<100; i++) {
    //     auto start = chrono::high_resolution_clock::now();

    //     timespec sleepduration;
    //     sleepduration.tv_nsec = 1000000L * (long)get_random_wait_time();
    //     sleepduration.tv_sec = sleepduration.tv_nsec / 1000000000L;
    //     sleepduration.tv_nsec %= 1000000000L;
    //     nanosleep(&sleepduration, NULL);

    //     auto stop = chrono::high_resolution_clock::now();
    //     auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start); 
    //     cout << "Time taken: " << duration.count() << " milliseconds" << endl;
    // }

    // struct rlimit rlim;
    // getrlimit(RLIMIT_NPROC, &rlim);
    int cpid = 0, cnt = 0;
    while (cpid != -1) {
        cpid = fork();
        if(cpid == 0) {
            sleep(5);
            exit(0);
        } else if(cpid == -1) {
            perror("Fork failed: ");
            printf("%d\n", errno);
            cout << cnt << endl;
        } else {
            cout << cpid << endl;
        }
        cnt++;
    }
    // cout << rlim.rlim_cur << endl;
    // cout << rlim.rlim_max << endl;
    return 0;
}
