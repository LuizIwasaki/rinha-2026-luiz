#include <iostream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <thread>
#include <vector>
#include <algorithm>

const int PORT = 8080;

void handle_client(int client_socket) {
    char buffer[4096] = {0};
    read(client_socket, buffer, 4096);
    
    std::string request(buffer);
    
    std::string response;
    if (request.find("GET /ready") == 0) {
        response = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
    } else if (request.find("POST /fraud-score") == 0) {
        // Mock rápido da lógica de vetor:
        std::string json_body = "{\"approved\": true, \"fraud_score\": 0.1}";
        response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " 
                   + std::to_string(json_body.length()) + "\r\n\r\n" + json_body;
    } else {
        response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n";
    }

    send(client_socket, response.c_str(), response.length(), 0);
    close(client_socket);
}

int main() {
    int server_fd;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);

    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        std::cerr << "Socket failed" << std::endl;
        return 1;
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        std::cerr << "setsockopt failed" << std::endl;
        return 1;
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        std::cerr << "Bind failed" << std::endl;
        return 1;
    }

    if (listen(server_fd, SOMAXCONN) < 0) {
        std::cerr << "Listen failed" << std::endl;
        return 1;
    }

    std::cout << "API C++ rodando na porta " << PORT << "..." << std::endl;

    while (true) {
        int client_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen);
        if (client_socket < 0) {
            continue;
        }
        
        // Em um cenário de altíssima performance, usaríamos Epoll em vez de thread por request
        std::thread(handle_client, client_socket).detach();
    }

    return 0;
}
