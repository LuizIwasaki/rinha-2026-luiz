// Rinha de Backend 2026 - Fraud Detection API (Multi-threaded + UDS + SSE2)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <emmintrin.h>

static constexpr int DIMS = 14;
static constexpr int KNN = 5;
static constexpr float THRESHOLD = 0.6f;
static constexpr int NPROBE = 10;
static constexpr int MAX_EVENTS = 1024;
static constexpr int BUF_SIZE = 4096;

static int32_t g_nrefs = 0, g_ncent = 0;
static float *g_centroids = nullptr, *g_vecs = nullptr;
static int32_t *g_offsets = nullptr;
static uint8_t *g_lbls = nullptr;

static float mcc_risk(const char* m, int len) {
    if (len != 4) return 0.5f;
    int v = (m[0]-'0')*1000+(m[1]-'0')*100+(m[2]-'0')*10+(m[3]-'0');
    switch(v) {
        case 5411: return 0.15f; case 5812: return 0.30f; case 5912: return 0.20f;
        case 5944: return 0.45f; case 7801: return 0.80f; case 7802: return 0.75f;
        case 7995: return 0.85f; case 4511: return 0.35f; case 5311: return 0.25f;
        case 5999: return 0.50f;
    }
    return 0.5f;
}

static void* map_file(const char* path, size_t& sz) {
    int fd = open(path, O_RDONLY); if (fd < 0) return nullptr;
    struct stat st; fstat(fd, &st); sz = st.st_size;
    void* p = mmap(nullptr, sz, PROT_READ, MAP_PRIVATE|MAP_POPULATE, fd, 0);
    close(fd); return p == MAP_FAILED ? nullptr : p;
}

static bool load_data() {
    const char* dir = getenv("DATA_DIR"); if (!dir) dir = "/data";
    char p[512]; size_t sz;
    snprintf(p, sizeof(p), "%s/centroids.bin", dir); auto* cm = (uint8_t*)map_file(p, sz); if (!cm) return false;
    g_ncent = *(int32_t*)cm; g_centroids = (float*)(cm + 8);
    snprintf(p, sizeof(p), "%s/offsets.bin", dir); auto* om = (uint8_t*)map_file(p, sz); if (!om) return false;
    g_offsets = (int32_t*)(om + 4);
    snprintf(p, sizeof(p), "%s/vectors.bin", dir); auto* vm = (uint8_t*)map_file(p, sz); if (!vm) return false;
    g_nrefs = *(int32_t*)vm; g_vecs = (float*)(vm + 4);
    snprintf(p, sizeof(p), "%s/labels.bin", dir); auto* lm = (uint8_t*)map_file(p, sz); if (!lm) return false;
    g_lbls = lm + 4;
    return true;
}

static inline float dist_sq_sse(const float* a, const float* b) {
    __m128 va = _mm_loadu_ps(a), vb = _mm_loadu_ps(b), d = _mm_sub_ps(va, vb), sum = _mm_mul_ps(d, d);
    va = _mm_loadu_ps(a + 4); vb = _mm_loadu_ps(b + 4); d = _mm_sub_ps(va, vb); sum = _mm_add_ps(sum, _mm_mul_ps(d, d));
    va = _mm_loadu_ps(a + 8); vb = _mm_loadu_ps(b + 8); d = _mm_sub_ps(va, vb); sum = _mm_add_ps(sum, _mm_mul_ps(d, d));
    sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 3, 0, 1)));
    sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 0, 3, 2)));
    float r; _mm_store_ss(&r, sum);
    float d12 = a[12]-b[12], d13 = a[13]-b[13]; return r + d12*d12 + d13*d13;
}

static const char* find_key(const char* js, const char* key) { return strstr(js, key); }
static double get_num(const char* js, const char* key) {
    const char* p = find_key(js, key); if (!p) return 0.0; p += strlen(key);
    while (*p && (*p=='"'||*p==':'||*p==' '||*p=='\t')) p++; return strtod(p, nullptr);
}
static bool get_bool(const char* js, const char* key) {
    const char* p = find_key(js, key); if (!p) return false; p += strlen(key);
    while (*p && (*p=='"'||*p==':'||*p==' ')) p++; return (*p == 't');
}
static bool is_known_merch(const char* js) {
    const char* merch = find_key(js, "\"merchant\""); if (!merch) return false;
    const char* id_p = find_key(merch, "\"id\""); if (!id_p) return false; id_p += 4;
    while (*id_p && *id_p != '"') id_p++; if (*id_p == '"') id_p++;
    const char* id_e = id_p; while (*id_e && *id_e != '"') id_e++;
    int id_len = id_e - id_p;
    const char* km = find_key(js, "\"known_merchants\""); if (!km) return false;
    km = strchr(km, '['); if (!km) return false;
    const char* s = km; while (*s && *s != ']') {
        s = strchr(s, '"'); if (!s) break; s++;
        const char* se = strchr(s, '"'); if (!se) break;
        if (se-s == id_len && memcmp(s, id_p, id_len) == 0) return true;
        s = se + 1;
    }
    return false;
}
static int get_hour(const char* js) {
    const char* p = find_key(js, "\"requested_at\""); if (!p) return 0;
    const char* t = strchr(p, 'T'); return t ? (t[1]-'0')*10+(t[2]-'0') : 0;
}
static int get_dow(const char* js) {
    const char* p = find_key(js, "\"requested_at\""); if (!p) return 0;
    while (*p && (*p=='"'||*p==':'||*p==' ')) p++; if (*p == '"') p++;
    int y=(p[0]-'0')*1000+(p[1]-'0')*100+(p[2]-'0')*10+(p[3]-'0');
    int m=(p[5]-'0')*10+(p[6]-'0'), d=(p[8]-'0')*10+(p[9]-'0');
    static int t[]={0,3,2,5,0,3,5,1,4,6,2,4}; if (m<3) y--;
    int dow=(y+y/4-y/100+y/400+t[m-1]+d)%7; return dow==0?6:dow-1;
}

static void vectorize(const char* js, float v[DIMS]) {
    float amt = get_num(js, "\"amount\""); v[0] = std::min(1.0f, amt/10000.0f);
    v[1] = std::min(1.0f, (float)get_num(js, "\"installments\"")/12.0f);
    float avg = get_num(js, "\"avg_amount\""); v[2] = avg>0 ? std::min(1.0f, (amt/avg)/10.0f) : 0.0f;
    v[3] = (float)get_hour(js)/23.0f; v[4] = (float)get_dow(js)/6.0f;
    const char* lt = find_key(js, "\"last_transaction\"");
    v[5] = (lt && !strstr(lt, "null")) ? 0.5f : -1.0f; // Simplified for speed
    v[6] = v[5] < 0 ? -1.0f : 0.2f;
    v[7] = std::min(1.0f, (float)get_num(js, "\"km_from_home\"")/1000.0f);
    v[8] = std::min(1.0f, (float)get_num(js, "\"tx_count_24h\"")/20.0f);
    v[9] = get_bool(js, "\"is_online\"") ? 1.0f : 0.0f;
    v[10] = get_bool(js, "\"card_present\"") ? 1.0f : 0.0f;
    v[11] = is_known_merch(js) ? 0.0f : 1.0f;
    const char* mcc_p = find_key(js, "\"mcc\"");
    if (mcc_p) { mcc_p+=5; while(*mcc_p && *mcc_p != '"') mcc_p++; if(*mcc_p=='"') v[12]=mcc_risk(mcc_p+1, 4); else v[12]=0.5f; } else v[12]=0.5f;
    float mavg = 0; const char* merch = find_key(js, "\"merchant\"");
    if (merch) { const char* ma = find_key(merch, "\"avg_amount\""); if (ma) mavg=strtof(ma+12, nullptr); }
    v[13] = std::min(1.0f, mavg/10000.0f);
}

static float knn_search(const float query[DIMS]) {
    struct CDist { float d; int i; }; CDist probe[NPROBE]; int np = 0;
    for (int c = 0; c < g_ncent; c++) {
        float d = dist_sq_sse(query, g_centroids + c * DIMS);
        if (np < NPROBE) { probe[np++] = {d, c}; std::push_heap(probe, probe+np, [](const CDist& a, const CDist& b){return a.d<b.d;}); }
        else if (d < probe[0].d) { std::pop_heap(probe, probe+NPROBE, [](const CDist& a, const CDist& b){return a.d<b.d;}); probe[NPROBE-1] = {d, c}; std::push_heap(probe, probe+NPROBE, [](const CDist& a, const CDist& b){return a.d<b.d;}); }
    }
    struct Neighbor { float d; int i; }; Neighbor top[KNN]; int nt = 0;
    for (int p = 0; p < np; p++) {
        int start = g_offsets[probe[p].i], end = g_offsets[probe[p].i+1];
        for (int i = start; i < end; i++) {
            float d = dist_sq_sse(query, g_vecs + i * DIMS);
            if (nt < KNN) { top[nt++] = {d, i}; std::push_heap(top, top+nt, [](const Neighbor& a, const Neighbor& b){return a.d<b.d;}); }
            else if (d < top[0].d) { std::pop_heap(top, top+KNN, [](const Neighbor& a, const Neighbor& b){return a.d<b.d;}); top[KNN-1] = {d, i}; std::push_heap(top, top+KNN, [](const Neighbor& a, const Neighbor& b){return a.d<b.d;}); }
        }
    }
    int f = 0; for (int i = 0; i < nt; i++) if (g_lbls[top[i].i] == 1) f++;
    return (float)f / (float)KNN;
}

struct Task { int fd; char buf[BUF_SIZE]; int len; };
static std::queue<Task> g_tasks; static std::mutex g_mtx; static std::condition_variable g_cv;

void worker() {
    char resp[BUF_SIZE];
    while (true) {
        Task t; { std::unique_lock<std::mutex> lock(g_mtx); g_cv.wait(lock, []{return !g_tasks.empty();}); t = g_tasks.front(); g_tasks.pop(); }
        if (t.fd == -1) break;
        if (memcmp(t.buf, "GET", 3) == 0 && strstr(t.buf, "/ready")) {
            int r = snprintf(resp, BUF_SIZE, "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            write(t.fd, resp, r);
        } else if (memcmp(t.buf, "POST", 4) == 0 && strstr(t.buf, "/fraud-score")) {
            const char* body = strstr(t.buf, "\r\n\r\n"); if (body) {
                float q[DIMS]; vectorize(body + 4, q); float s = knn_search(q);
                char json[128]; int jl = snprintf(json, sizeof(json), "{\"approved\":%s,\"fraud_score\":%.1f}", s < THRESHOLD ? "true" : "false", s);
                int r = snprintf(resp, BUF_SIZE, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s", jl, json);
                write(t.fd, resp, r);
            }
        }
        close(t.fd);
    }
}

int main() {
    if (!load_data()) return 1;
    std::thread t1(worker), t2(worker);
    const char* sock_path = getenv("SOCKET_PATH");
    int sfd;
    if (sock_path) {
        sfd = socket(AF_UNIX, SOCK_STREAM, 0); unlink(sock_path);
        struct sockaddr_un addr = {}; addr.sun_family = AF_UNIX; strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path)-1);
        bind(sfd, (struct sockaddr*)&addr, sizeof(addr)); chmod(sock_path, 0666);
    } else {
        sfd = socket(AF_INET, SOCK_STREAM, 0); int opt = 1; setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR|SO_REUSEPORT, &opt, sizeof(opt));
        struct sockaddr_in addr = {}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(8080);
        bind(sfd, (struct sockaddr*)&addr, sizeof(addr));
    }
    listen(sfd, SOMAXCONN); fcntl(sfd, F_SETFL, O_NONBLOCK);
    int epfd = epoll_create1(0); struct epoll_event ev, events[MAX_EVENTS]; ev.events = EPOLLIN; ev.data.fd = sfd; epoll_ctl(epfd, EPOLL_CTL_ADD, sfd, &ev);
    while (true) {
        int n = epoll_wait(epfd, events, MAX_EVENTS, -1);
        for (int i = 0; i < n; i++) {
            if (events[i].data.fd == sfd) {
                while (true) {
                    int cfd = accept4(sfd, nullptr, nullptr, SOCK_NONBLOCK); if (cfd < 0) break;
                    ev.events = EPOLLIN | EPOLLET; ev.data.fd = cfd; epoll_ctl(epfd, EPOLL_CTL_ADD, cfd, &ev);
                }
            } else {
                int fd = events[i].data.fd; Task t; t.fd = fd; t.len = read(fd, t.buf, BUF_SIZE-1);
                if (t.len > 0) { t.buf[t.len] = 0; std::lock_guard<std::mutex> lock(g_mtx); g_tasks.push(t); g_cv.notify_one(); epoll_ctl(epfd, EPOLL_CTL_DEL, fd, nullptr); }
                else if (t.len == 0 || (t.len < 0 && errno != EAGAIN)) { epoll_ctl(epfd, EPOLL_CTL_DEL, fd, nullptr); close(fd); }
            }
        }
    }
    return 0;
}
