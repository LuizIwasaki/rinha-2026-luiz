// Rinha 2026 - Epoll + HTTP Keep-Alive + AVX2 + NPROBE=1
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <immintrin.h>

static constexpr int DIMS=14, KNN=5, NPROBE=1, BUF_SIZE=4096, MAX_EV=512;
static constexpr float THRESHOLD=0.6f;
static int32_t g_nrefs=0,g_ncent=0;
static float *g_cent=nullptr,*g_vecs=nullptr;
static int32_t *g_off=nullptr;
static uint8_t *g_lbl=nullptr;

static void* mf(const char* p,size_t& sz){int fd=open(p,O_RDONLY);if(fd<0)return nullptr;struct stat st;fstat(fd,&st);sz=st.st_size;void*m=mmap(nullptr,sz,PROT_READ,MAP_PRIVATE|MAP_POPULATE,fd,0);close(fd);return m==MAP_FAILED?nullptr:m;}
static bool load(){
    const char*dir=getenv("DATA_DIR");if(!dir)dir="/data";char p[512];size_t sz;
    snprintf(p,512,"%s/centroids.bin",dir);auto*cm=(uint8_t*)mf(p,sz);if(!cm)return false;g_ncent=*(int32_t*)cm;g_cent=(float*)(cm+8);
    snprintf(p,512,"%s/offsets.bin",dir);auto*om=(uint8_t*)mf(p,sz);if(!om)return false;g_off=(int32_t*)(om+4);
    snprintf(p,512,"%s/vectors.bin",dir);auto*vm=(uint8_t*)mf(p,sz);if(!vm)return false;g_nrefs=*(int32_t*)vm;g_vecs=(float*)(vm+4);
    snprintf(p,512,"%s/labels.bin",dir);auto*lm=(uint8_t*)mf(p,sz);if(!lm)return false;g_lbl=lm+4;
    return true;
}
static inline float dsse(const float* a, const float* b){
    __m256 v0 = _mm256_sub_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b));
    __m256 sq0 = _mm256_mul_ps(v0, v0);
    __m128 v1 = _mm_sub_ps(_mm_loadu_ps(a+8), _mm_loadu_ps(b+8));
    __m128 sq1 = _mm_mul_ps(v1, v1);
    __m128 h1 = _mm_add_ps(_mm256_castps256_ps128(sq0), _mm256_extractf128_ps(sq0, 1));
    __m128 s = _mm_add_ps(h1, sq1);
    s = _mm_add_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(2, 3, 0, 1)));
    s = _mm_add_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 0, 3, 2)));
    float r; _mm_store_ss(&r, s);
    float x12 = a[12]-b[12], x13 = a[13]-b[13];
    return r + x12*x12 + x13*x13;
}
static float mcc_r(const char*m){
    if(!m)return 0.5f;int v=(m[0]-'0')*1000+(m[1]-'0')*100+(m[2]-'0')*10+(m[3]-'0');
    switch(v){case 5411:return 0.15f;case 5812:return 0.30f;case 5912:return 0.20f;case 5944:return 0.45f;case 7801:return 0.80f;case 7802:return 0.75f;case 7995:return 0.85f;case 4511:return 0.35f;case 5311:return 0.25f;case 5999:return 0.50f;}return 0.5f;
}
static double gn(const char*js,const char*k){const char*p=strstr(js,k);if(!p)return 0;p+=strlen(k);while(*p&&(*p=='"'||*p==':'||*p==' '||*p=='\t'))p++;return strtod(p,nullptr);}
static bool gb(const char*js,const char*k){const char*p=strstr(js,k);if(!p)return false;p+=strlen(k);while(*p&&(*p=='"'||*p==':'||*p==' '))p++;return *p=='t';}
static bool ikm(const char*js){
    const char*m=strstr(js,"\"merchant\"");if(!m)return false;const char*i=strstr(m,"\"id\"");if(!i)return false;i+=4;while(*i&&*i!='"')i++;if(*i=='"')i++;
    const char*e=i;while(*e&&*e!='"')e++;int l=e-i;const char*k=strstr(js,"\"known_merchants\"");if(!k)return false;k=strchr(k,'[');if(!k)return false;
    const char*s=k;while(*s&&*s!=']'){s=strchr(s,'"');if(!s)break;s++;const char*se=strchr(s,'"');if(!se)break;if(se-s==l&&memcmp(s,i,l)==0)return true;s=se+1;}return false;
}
static int gh(const char*js){const char*p=strstr(js,"\"requested_at\"");if(!p)return 0;const char*t=strchr(p,'T');return t?(t[1]-'0')*10+(t[2]-'0'):0;}
static int gd(const char*js){
    const char*p=strstr(js,"\"requested_at\"");if(!p)return 0;p+=14;while(*p&&(*p=='"'||*p==':'||*p==' '))p++;if(*p=='"')p++;
    int y=(p[0]-'0')*1000+(p[1]-'0')*100+(p[2]-'0')*10+(p[3]-'0'),m=(p[5]-'0')*10+(p[6]-'0'),d=(p[8]-'0')*10+(p[9]-'0');
    static int t[]={0,3,2,5,0,3,5,1,4,6,2,4};if(m<3)y--;int dow=(y+y/4-y/100+y/400+t[m-1]+d)%7;return dow==0?6:dow-1;
}
static double gms(const char*js){
    const char*lt=strstr(js,"\"last_transaction\"");if(!lt)return -1;const char*ts=strstr(lt,"\"timestamp\"");if(!ts)return -1;
    ts+=11;while(*ts&&(*ts=='"'||*ts==':'||*ts==' '))ts++;if(*ts=='"')ts++;
    const char*ra=strstr(js,"\"requested_at\"");if(!ra)return -1;ra+=14;while(*ra&&(*ra=='"'||*ra==':'||*ra==' '))ra++;if(*ra=='"')ra++;
    auto pa=[](const char*s,int&Y,int&M,int&D,int&h,int&m,int&sec){Y=(s[0]-'0')*1000+(s[1]-'0')*100+(s[2]-'0')*10+(s[3]-'0');M=(s[5]-'0')*10+(s[6]-'0');D=(s[8]-'0')*10+(s[9]-'0');h=(s[11]-'0')*10+(s[12]-'0');m=(s[14]-'0')*10+(s[15]-'0');sec=(s[17]-'0')*10+(s[18]-'0');};
    int y1,m1,d1,h1,mi1,s1,y2,m2,d2,h2,mi2,s2;pa(ra,y1,m1,d1,h1,mi1,s1);pa(ts,y2,m2,d2,h2,mi2,s2);
    auto tm=[](int y,int m,int d,int h,int mi,int)->long long{long long ds=(long long)y*365+y/4-y/100+y/400;int md[]={0,31,59,90,120,151,181,212,243,273,304,334};ds+=md[m-1]+d;if(m>2&&((y%4==0&&y%100!=0)||y%400==0))ds++;return ds*1440LL+h*60LL+mi;};
    return(double)(tm(y1,m1,d1,h1,mi1,s1)-tm(y2,m2,d2,h2,mi2,s2));
}
static inline float cl(float x){return x<0?0:(x>1?1:x);}
static void vec(const char*js,float v[DIMS]){
    float a=gn(js,"\"amount\"");v[0]=cl(a/10000);v[1]=cl(gn(js,"\"installments\"")/12);float av=gn(js,"\"avg_amount\"");v[2]=av>0?cl((a/av)/10):0;
    v[3]=(float)gh(js)/23;v[4]=(float)gd(js)/6;
    const char*lt=strstr(js,"\"last_transaction\"");bool ln=true;if(lt){lt+=18;while(*lt==' '||*lt==':')lt++;ln=(*lt=='n');}
    if(ln){v[5]=-1;v[6]=-1;}else{v[5]=cl(gms(js)/1440);v[6]=cl(gn(js,"\"km_from_current\"")/1000);}
    v[7]=cl(gn(js,"\"km_from_home\"")/1000);v[8]=cl(gn(js,"\"tx_count_24h\"")/20);v[9]=gb(js,"\"is_online\"")?1:0;v[10]=gb(js,"\"card_present\"")?1:0;v[11]=ikm(js)?0:1;
    const char*mc=strstr(js,"\"mcc\"");if(mc){mc+=5;while(*mc&&*mc!='"')mc++;v[12]=(*mc=='"')?mcc_r(mc+1):0.5f;}else v[12]=0.5f;
    float ma=0;const char*me=strstr(js,"\"merchant\"");if(me){const char*a2=strstr(me,"\"avg_amount\"");if(a2){a2+=12;while(*a2&&(*a2=='"'||*a2==':'||*a2==' '))a2++;ma=strtof(a2,nullptr);}}v[13]=cl(ma/10000);
}
static float knn(const float q[DIMS]){
    int best_c=-1;float best_d=1e30f;
    for(int c=0;c<g_ncent;c++){float d=dsse(q,g_cent+c*DIMS);if(d<best_d){best_d=d;best_c=c;}}
    struct N{float d;int i;};N top[KNN];int nt=0;
    int s=g_off[best_c],e=g_off[best_c+1];
    for(int i=s;i<e;i++){
        float d=dsse(q,g_vecs+i*DIMS);
        if(nt<KNN){top[nt++]={d,i};if(nt==KNN){for(int a=1;a<KNN;a++){N tmp=top[a];int b=a-1;while(b>=0&&top[b].d<tmp.d){top[b+1]=top[b];b--;}top[b+1]=tmp;}}}
        else if(d<top[0].d){int b=1;while(b<KNN&&top[b].d>d){top[b-1]=top[b];b++;}top[b-1]={d,i};}
    }
    int f=0;for(int i=0;i<nt;i++)if(g_lbl[top[i].i]==1)f++;return(float)f/KNN;
}

// Connection buffer - reduced to 2048 entries to save memory
struct CB{char buf[BUF_SIZE];int len;};
static CB g_cb[4096];

// Find end of current HTTP request, returns pointer past body or nullptr
static const char* find_request_end(const char* buf, int len) {
    const char* hdr_end = strstr(buf, "\r\n\r\n");
    if (!hdr_end) return nullptr;
    const char* body = hdr_end + 4;
    if (memcmp(buf, "GET", 3) == 0) return body; // GET has no body
    // POST - check Content-Length
    const char* cl = strstr(buf, "Content-Length:");
    if (!cl) cl = strstr(buf, "content-length:");
    if (!cl) return body;
    int clen = atoi(cl + 15);
    int body_offset = body - buf;
    if (len - body_offset >= clen) return body + clen;
    return nullptr; // body incomplete
}

static int make_response(const char* req, char* resp) {
    if (memcmp(req, "GET", 3) == 0 && strstr(req, "/ready"))
        return snprintf(resp, BUF_SIZE, "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
    if (memcmp(req, "POST", 4) == 0 && strstr(req, "/fraud-score")) {
        const char* b = strstr(req, "\r\n\r\n");
        if (!b) return 0;
        b += 4;
        float q[DIMS]; vec(b, q); float s = knn(q);
        char j[128];
        int jl = snprintf(j, 128, "{\"approved\":%s,\"fraud_score\":%.1f}", s < THRESHOLD ? "true" : "false", s);
        return snprintf(resp, BUF_SIZE, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s", jl, j);
    }
    return snprintf(resp, BUF_SIZE, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
}

int main(){
    int port=8080; const char*pe=getenv("PORT"); if(pe) port=atoi(pe);
    if(!load()) return 1;
    fprintf(stderr, "Loaded %d refs, %d centroids, port %d\n", g_nrefs, g_ncent, port);

    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(port);
    bind(sfd, (struct sockaddr*)&addr, sizeof(addr));
    listen(sfd, SOMAXCONN);
    fcntl(sfd, F_SETFL, O_NONBLOCK);

    int epfd = epoll_create1(0);
    struct epoll_event ev, events[MAX_EV];
    ev.events = EPOLLIN; ev.data.fd = sfd;
    epoll_ctl(epfd, EPOLL_CTL_ADD, sfd, &ev);

    char resp[BUF_SIZE];

    while (true) {
        int n = epoll_wait(epfd, events, MAX_EV, -1);
        for (int i = 0; i < n; i++) {
            int fd = events[i].data.fd;

            if (fd == sfd) {
                // Accept all pending connections
                while (true) {
                    int c = accept4(sfd, nullptr, nullptr, SOCK_NONBLOCK);
                    if (c < 0) break;
                    if (c >= 4096) { close(c); continue; } // fd too high
                    int nd = 1;
                    setsockopt(c, IPPROTO_TCP, TCP_NODELAY, &nd, sizeof(nd));
                    g_cb[c].len = 0;
                    ev.events = EPOLLIN; // Level-triggered for keep-alive
                    ev.data.fd = c;
                    epoll_ctl(epfd, EPOLL_CTL_ADD, c, &ev);
                }
                continue;
            }

            // Data event on client fd
            if (events[i].events & (EPOLLHUP | EPOLLERR)) {
                epoll_ctl(epfd, EPOLL_CTL_DEL, fd, nullptr);
                close(fd);
                continue;
            }

            CB& cb = g_cb[fd];
            int r = read(fd, cb.buf + cb.len, BUF_SIZE - cb.len - 1);
            if (r <= 0) {
                epoll_ctl(epfd, EPOLL_CTL_DEL, fd, nullptr);
                close(fd);
                continue;
            }
            cb.len += r;
            cb.buf[cb.len] = 0;

            // Try to process complete request(s)
            const char* req_end = find_request_end(cb.buf, cb.len);
            if (!req_end) {
                if (cb.len >= BUF_SIZE - 1) { // Buffer full, drop
                    epoll_ctl(epfd, EPOLL_CTL_DEL, fd, nullptr);
                    close(fd);
                }
                continue;
            }

            // Process the request
            int rl = make_response(cb.buf, resp);
            if (rl > 0) {
                int sent = 0;
                while (sent < rl) {
                    int w = write(fd, resp + sent, rl - sent);
                    if (w <= 0) break;
                    sent += w;
                }
            }

            // Keep-alive: shift remaining data and reset for next request
            int consumed = req_end - cb.buf;
            int remaining = cb.len - consumed;
            if (remaining > 0) {
                memmove(cb.buf, req_end, remaining);
                cb.len = remaining;
            } else {
                cb.len = 0;
            }
            // Connection stays open - HAProxy will reuse it
        }
    }
}
