// Rinha de Backend 2026 - Fraud Detection (Epoll + Fixed ThreadPool + SSE2)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <functional>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <emmintrin.h>

static constexpr int DIMS = 14;
static constexpr int KNN = 5;
static constexpr float THRESHOLD = 0.6f;
static constexpr int NPROBE = 3;
static constexpr int BUF_SIZE = 4096;
static constexpr int POOL_SIZE = 2;

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
    snprintf(p, sizeof(p), "%s/centroids.bin", dir);
    auto* cm = (uint8_t*)map_file(p, sz); if (!cm) return false;
    g_ncent = *(int32_t*)cm; g_centroids = (float*)(cm + 8);
    snprintf(p, sizeof(p), "%s/offsets.bin", dir);
    auto* om = (uint8_t*)map_file(p, sz); if (!om) return false;
    g_offsets = (int32_t*)(om + 4);
    snprintf(p, sizeof(p), "%s/vectors.bin", dir);
    auto* vm = (uint8_t*)map_file(p, sz); if (!vm) return false;
    g_nrefs = *(int32_t*)vm; g_vecs = (float*)(vm + 4);
    snprintf(p, sizeof(p), "%s/labels.bin", dir);
    auto* lm = (uint8_t*)map_file(p, sz); if (!lm) return false;
    g_lbls = lm + 4;
    return true;
}

static inline float dist_sq_sse(const float* __restrict__ a, const float* __restrict__ b) {
    __m128 d0 = _mm_sub_ps(_mm_loadu_ps(a), _mm_loadu_ps(b));
    __m128 d1 = _mm_sub_ps(_mm_loadu_ps(a+4), _mm_loadu_ps(b+4));
    __m128 d2 = _mm_sub_ps(_mm_loadu_ps(a+8), _mm_loadu_ps(b+8));
    __m128 sum = _mm_add_ps(_mm_add_ps(_mm_mul_ps(d0,d0), _mm_mul_ps(d1,d1)), _mm_mul_ps(d2,d2));
    sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2,3,0,1)));
    sum = _mm_add_ps(sum, _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1,0,3,2)));
    float r; _mm_store_ss(&r, sum);
    float d12 = a[12]-b[12], d13 = a[13]-b[13];
    return r + d12*d12 + d13*d13;
}

static double get_num(const char* js, const char* key) {
    const char* p = strstr(js, key); if (!p) return 0;
    p += strlen(key); while (*p && (*p=='"'||*p==':'||*p==' '||*p=='\t')) p++;
    return strtod(p, nullptr);
}
static bool get_bool(const char* js, const char* key) {
    const char* p = strstr(js, key); if (!p) return false;
    p += strlen(key); while (*p && (*p=='"'||*p==':'||*p==' ')) p++;
    return *p == 't';
}
static bool is_known_merch(const char* js) {
    const char* merch = strstr(js, "\"merchant\""); if (!merch) return false;
    const char* id_p = strstr(merch, "\"id\""); if (!id_p) return false;
    id_p += 4; while (*id_p && *id_p != '"') id_p++; if (*id_p=='"') id_p++;
    const char* id_e = id_p; while (*id_e && *id_e != '"') id_e++;
    int id_len = id_e - id_p;
    const char* km = strstr(js, "\"known_merchants\""); if (!km) return false;
    km = strchr(km, '['); if (!km) return false;
    const char* s = km;
    while (*s && *s != ']') {
        s = strchr(s, '"'); if (!s) break; s++;
        const char* se = strchr(s, '"'); if (!se) break;
        if (se-s == id_len && memcmp(s, id_p, id_len) == 0) return true;
        s = se + 1;
    }
    return false;
}
static int get_hour(const char* js) {
    const char* p = strstr(js, "\"requested_at\""); if (!p) return 0;
    const char* t = strchr(p, 'T'); return t ? (t[1]-'0')*10+(t[2]-'0') : 0;
}
static int get_dow(const char* js) {
    const char* p = strstr(js, "\"requested_at\""); if (!p) return 0;
    p += 14; while (*p && (*p=='"'||*p==':'||*p==' ')) p++; if (*p=='"') p++;
    int y=(p[0]-'0')*1000+(p[1]-'0')*100+(p[2]-'0')*10+(p[3]-'0');
    int m=(p[5]-'0')*10+(p[6]-'0'), d=(p[8]-'0')*10+(p[9]-'0');
    static int t[]={0,3,2,5,0,3,5,1,4,6,2,4}; if (m<3) y--;
    int dow=(y+y/4-y/100+y/400+t[m-1]+d)%7; return dow==0?6:dow-1;
}
static double get_minutes_since(const char* js) {
    const char* lt = strstr(js, "\"last_transaction\""); if (!lt) return -1;
    const char* ts = strstr(lt, "\"timestamp\""); if (!ts) return -1;
    ts += 11; while (*ts && (*ts=='"'||*ts==':'||*ts==' ')) ts++; if (*ts=='"') ts++;
    const char* ra = strstr(js, "\"requested_at\""); if (!ra) return -1;
    ra += 14; while (*ra && (*ra=='"'||*ra==':'||*ra==' ')) ra++; if (*ra=='"') ra++;
    auto parse = [](const char* s, int& Y, int& M, int& D, int& h, int& m, int& sec) {
        Y=(s[0]-'0')*1000+(s[1]-'0')*100+(s[2]-'0')*10+(s[3]-'0');
        M=(s[5]-'0')*10+(s[6]-'0'); D=(s[8]-'0')*10+(s[9]-'0');
        h=(s[11]-'0')*10+(s[12]-'0'); m=(s[14]-'0')*10+(s[15]-'0'); sec=(s[17]-'0')*10+(s[18]-'0');
    };
    int y1,m1,d1,h1,mi1,s1,y2,m2,d2,h2,mi2,s2;
    parse(ra, y1,m1,d1,h1,mi1,s1); parse(ts, y2,m2,d2,h2,mi2,s2);
    auto to_min=[](int y,int m,int d,int h,int mi,int)->long long{
        long long days=(long long)y*365+y/4-y/100+y/400;
        int md[]={0,31,59,90,120,151,181,212,243,273,304,334};
        days+=md[m-1]+d; if(m>2&&((y%4==0&&y%100!=0)||y%400==0)) days++;
        return days*1440LL+h*60LL+mi;
    };
    return (double)(to_min(y1,m1,d1,h1,mi1,s1)-to_min(y2,m2,d2,h2,mi2,s2));
}
static inline float clamp01(float x) { return x<0?0:(x>1?1:x); }

static void vectorize(const char* js, float v[DIMS]) {
    float amt = get_num(js, "\"amount\"");
    v[0] = clamp01(amt/10000.0f);
    v[1] = clamp01((float)get_num(js, "\"installments\"")/12.0f);
    float avg = get_num(js, "\"avg_amount\"");
    v[2] = avg > 0 ? clamp01((amt/avg)/10.0f) : 0.0f;
    v[3] = (float)get_hour(js)/23.0f;
    v[4] = (float)get_dow(js)/6.0f;
    const char* lt = strstr(js, "\"last_transaction\"");
    bool lt_null = true;
    if (lt) { lt += 18; while(*lt==' '||*lt==':') lt++; lt_null = (*lt=='n'); }
    if (lt_null) { v[5] = -1.0f; v[6] = -1.0f; }
    else {
        double mins = get_minutes_since(js);
        v[5] = clamp01((float)(mins/1440.0));
        v[6] = clamp01((float)get_num(js, "\"km_from_current\"")/1000.0f);
    }
    v[7] = clamp01((float)get_num(js, "\"km_from_home\"")/1000.0f);
    v[8] = clamp01((float)get_num(js, "\"tx_count_24h\"")/20.0f);
    v[9] = get_bool(js, "\"is_online\"") ? 1.0f : 0.0f;
    v[10] = get_bool(js, "\"card_present\"") ? 1.0f : 0.0f;
    v[11] = is_known_merch(js) ? 0.0f : 1.0f;
    const char* mcc_p = strstr(js, "\"mcc\"");
    if (mcc_p) { mcc_p+=5; while(*mcc_p&&*mcc_p!='"') mcc_p++; if(*mcc_p=='"') v[12]=mcc_risk(mcc_p+1,4); else v[12]=0.5f; } else v[12]=0.5f;
    float mavg = 0; const char* merch = strstr(js, "\"merchant\"");
    if (merch) { const char* ma = strstr(merch, "\"avg_amount\""); if (ma) { ma+=12; while(*ma&&(*ma=='"'||*ma==':'||*ma==' '))ma++; mavg=strtof(ma,nullptr); } }
    v[13] = clamp01(mavg/10000.0f);
}

static float knn_search(const float query[DIMS]) {
    struct CDist { float d; int i; };
    CDist probe[NPROBE]; int np = 0;
    for (int c = 0; c < g_ncent; c++) {
        float d = dist_sq_sse(query, g_centroids + c * DIMS);
        if (np < NPROBE) {
            probe[np++] = {d, c};
            if (np == NPROBE) for (int j = NPROBE/2-1; j >= 0; j--) {
                int k=j; while(2*k+1<NPROBE){int ch=2*k+1;if(ch+1<NPROBE&&probe[ch+1].d>probe[ch].d)ch++;if(probe[k].d>=probe[ch].d)break;CDist t=probe[k];probe[k]=probe[ch];probe[ch]=t;k=ch;}
            }
        } else if (d < probe[0].d) {
            probe[0]={d,c}; int k=0; while(2*k+1<NPROBE){int ch=2*k+1;if(ch+1<NPROBE&&probe[ch+1].d>probe[ch].d)ch++;if(probe[k].d>=probe[ch].d)break;CDist t=probe[k];probe[k]=probe[ch];probe[ch]=t;k=ch;}
        }
    }
    struct Neighbor { float d; int i; }; Neighbor top[KNN]; int nt = 0;
    for (int p = 0; p < np; p++) {
        int start = g_offsets[probe[p].i], end = g_offsets[probe[p].i+1];
        for (int i = start; i < end; i++) {
            float d = dist_sq_sse(query, g_vecs + i * DIMS);
            if (nt < KNN) { top[nt++]={d,i}; if(nt==KNN) for(int j=KNN/2-1;j>=0;j--){int k=j;while(2*k+1<KNN){int ch=2*k+1;if(ch+1<KNN&&top[ch+1].d>top[ch].d)ch++;if(top[k].d>=top[ch].d)break;Neighbor t=top[k];top[k]=top[ch];top[ch]=t;k=ch;}} }
            else if (d < top[0].d) { top[0]={d,i}; int k=0; while(2*k+1<KNN){int ch=2*k+1;if(ch+1<KNN&&top[ch+1].d>top[ch].d)ch++;if(top[k].d>=top[ch].d)break;Neighbor t=top[k];top[k]=top[ch];top[ch]=t;k=ch;} }
        }
    }
    int f = 0; for (int i = 0; i < nt; i++) if (g_lbls[top[i].i] == 1) f++;
    return (float)f / (float)KNN;
}

// Thread pool
class ThreadPool {
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop = false;
public:
    ThreadPool(int n) {
        for (int i = 0; i < n; i++)
            workers.emplace_back([this]{
                while (true) {
                    std::function<void()> task;
                    { std::unique_lock<std::mutex> lock(mtx); cv.wait(lock, [this]{return stop||!tasks.empty();}); if(stop&&tasks.empty()) return; task=std::move(tasks.front()); tasks.pop(); }
                    task();
                }
            });
    }
    void submit(std::function<void()> task) {
        { std::lock_guard<std::mutex> lock(mtx); tasks.push(std::move(task)); }
        cv.notify_one();
    }
    ~ThreadPool() { {std::lock_guard<std::mutex> lock(mtx); stop=true;} cv.notify_all(); for(auto& w:workers) w.join(); }
};

static void handle_fd(int fd) {
    char buf[BUF_SIZE], resp[BUF_SIZE];
    int total = 0;
    while (total < BUF_SIZE - 1) {
        int r = read(fd, buf + total, BUF_SIZE - 1 - total);
        if (r <= 0) { if (r == 0 || errno != EAGAIN) break; continue; }
        total += r; buf[total] = 0;
        if (strstr(buf, "\r\n\r\n")) break;
    }
    if (total <= 0) { close(fd); return; }
    buf[total] = 0;
    int rlen = 0;
    if (memcmp(buf, "GET", 3) == 0 && strstr(buf, "/ready")) {
        rlen = snprintf(resp, BUF_SIZE, "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    } else if (memcmp(buf, "POST", 4) == 0 && strstr(buf, "/fraud-score")) {
        const char* hdr_end = strstr(buf, "\r\n\r\n");
        if (hdr_end) {
            const char* cl = strstr(buf, "Content-Length:");
            if (!cl) cl = strstr(buf, "content-length:");
            if (cl) {
                int clen = atoi(cl + 15), bs = (hdr_end+4)-buf;
                while (total - bs < clen && total < BUF_SIZE-1) {
                    int r = read(fd, buf+total, BUF_SIZE-1-total);
                    if (r <= 0) break; total += r;
                }
            }
            buf[total] = 0;
            float q[DIMS]; vectorize(hdr_end+4, q);
            float score = knn_search(q);
            char json[128]; int jl = snprintf(json, sizeof(json), "{\"approved\":%s,\"fraud_score\":%.1f}", score<THRESHOLD?"true":"false", score);
            rlen = snprintf(resp, BUF_SIZE, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s", jl, json);
        }
    } else {
        rlen = snprintf(resp, BUF_SIZE, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    }
    if (rlen > 0) { int s=0; while(s<rlen){int w=write(fd,resp+s,rlen-s);if(w<=0)break;s+=w;} }
    close(fd);
}

int main() {
    int port = 8080; const char* pe = getenv("PORT"); if (pe) port = atoi(pe);
    if (!load_data()) { fprintf(stderr, "Failed to load data\n"); return 1; }
    fprintf(stderr, "Data loaded, port %d\n", port);

    ThreadPool pool(POOL_SIZE);

    int sfd = socket(AF_INET, SOCK_STREAM, 0); int opt = 1;
    setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    setsockopt(sfd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
    struct sockaddr_in addr = {}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(port);
    if (bind(sfd, (struct sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    listen(sfd, SOMAXCONN);

    while (true) {
        int cfd = accept(sfd, nullptr, nullptr);
        if (cfd < 0) continue;
        int nd = 1; setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &nd, sizeof(nd));
        pool.submit([cfd]{ handle_fd(cfd); });
    }
    return 0;
}
