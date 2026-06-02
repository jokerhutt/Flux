// TLS Echo Client
// Connects to server at 127.0.0.1:8443 and sends messages over TLS

#import "standard.fx";
#import "socket_object_raw.fx";
#import "tls.fx";

using standard::io::console,
      standard::io::sockets,
      standard::strings,
      standard::tls;

// L"localhost" as UTF-16LE for SNI / target name
byte[20] SERVER_NAME_W = [
    0x6C,0x00,0x6F,0x00,0x63,0x00,0x61,0x00,  // l o c a
    0x6C,0x00,0x68,0x00,0x6F,0x00,0x73,0x00,  // l h o s
    0x74,0x00,0x00,0x00                         // t \0
];

def main() -> int
{
    // Initialize Winsock
    int init_result = init();
    if (init_result != 0)
    {
        print("Failed to initialize Winsock\n\0");
        return 1;
    };

    print("=== TLS Echo Client ===\n\0");
    print("Connecting to server at 127.0.0.1:8443...\n\0");

    // Create TCP socket and connect
    socket client_socket(socket_type.TCP);
    client_socket.fd = tcp_socket();

    if (!client_socket.is_open())
    {
        print("Failed to create socket\n\0");
        cleanup();
        return 1;
    };

    if (!client_socket.connect("127.0.0.1\0", (i16)8443))
    {
        print("Failed to connect to server\n\0");
        client_socket.close();
        cleanup();
        return 1;
    };

    print("TCP connection established\n\0");

    // Set up TLS connection over the connected socket
    TlsConn tls(client_socket.fd);

    // Acquire client credentials (no client certificate — server-auth only)
    if (!conn::client_create_cred(@tls, (void*)0))
    {
        print("Failed to acquire TLS credentials\n\0");
        client_socket.close();
        cleanup();
        return 1;
    };

    print("Performing TLS handshake...\n\0");

    // Perform the TLS handshake
    // Pass server name for SNI and certificate validation
    if (!conn::client_handshake(@tls, @SERVER_NAME_W[0]))
    {
        print("TLS handshake failed (error \0");
        print(tls.error_state);
        print(")\n\0");
        tls.__exit();
        client_socket.close();
        cleanup();
        return 1;
    };

    print("TLS handshake complete!\n\0");

    // Buffer for sending and receiving data
    byte* send_buffer = fmalloc(1024);
    byte* recv_buffer = fmalloc(1024);
    defer ffree((u64)send_buffer);
    defer ffree((u64)recv_buffer);

    // Send some test messages
    int message_count = 3;
    int i;
    byte* msg;
    int msg_len;

    byte*[3] messages;
    messages[0] = "Hello, Server!\0";
    messages[1] = "This is message 2\0";
    messages[2] = "Final message from client\0";

    while (i < message_count)
    {
        print("\n--- Message \0");
        print(i + 1);
        print(" ---\n\0");

        msg = messages[i];

        msg_len = 0;
        while (msg[msg_len] != 0)
        {
            send_buffer[msg_len] = msg[msg_len];
            msg_len = msg_len + 1;
        };

        print("Sending: \0");
        print(msg);
        print("\n\0");

        // Send over TLS
        int bytes_sent = conn::tls_send(@tls, send_buffer, (u32)msg_len);

        if (bytes_sent < 0)
        {
            print("Failed to send message\n\0");
            break;
        };

        print("Sent \0");
        print(bytes_sent);
        print(" bytes\n\0");

        // Receive echo response over TLS
        int bytes_received = conn::tls_recv(@tls, recv_buffer, 1024u);

        if (bytes_received <= 0)
        {
            print("Server disconnected or error occurred\n\0");
            break;
        };

        recv_buffer[bytes_received] = '\0';

        print("Received echo (\0");
        print(bytes_received);
        print(" bytes): \0");
        print(recv_buffer);
        print("\n\0");

        i = i + 1;
    };

    // Graceful TLS shutdown before closing socket
    conn::tls_shutdown(@tls);
    tls.__exit();

    client_socket.close();
    cleanup();

    print("\nClient finished\n\0");
    return 0;
};
