// TLS Echo Server
// Listens on port 8443 and echoes back any messages received over TLS
// Expects a DER-encoded server certificate in "server.cer"
// and a corresponding BCRYPT_RSAFULLPRIVATEBLOB in "server.key"

#import "standard.fx";
#import "socket_object_raw.fx";
#import "tls.fx";

using standard::io::console,
      standard::io::sockets,
      standard::strings;

// Seek to end to get size
// fseek / ftell via libc — declared here locally
extern { stdcall !! fseek(byte*, long, int) -> int, ftell(byte*) -> long, fread(void*, size_t, size_t, byte*) -> size_t; };

// Load a binary file from disk into a heap buffer.
// Returns the buffer pointer; sets *out_len.
// Returns void* 0 on failure.
def load_file(byte* path, u32* out_len) -> byte*
{
    byte* f = fopen(path, "rb\0");
    if (f == (byte*)0) { return (byte*)0; };


    fseek(f, 0, 2);    // SEEK_END = 2
    long sz = ftell(f);
    fseek(f, 0, 0);    // SEEK_SET = 0

    if (sz <= 0) { fclose(f); return (byte*)0; };

    byte* buf = fmalloc((size_t)sz);
    if (buf == (byte*)0) { fclose(f); return (byte*)0; };

    fread(buf, 1, (size_t)sz, f);
    fclose(f);

    *out_len = (u32)sz;
    return buf;
};

def main() -> int
{
    // Initialize Winsock
    int init_result = init();
    if (init_result != 0)
    {
        print("Failed to initialize Winsock\n\0");
        return 1;
    };

    print("=== TLS Echo Server ===\n\0");

    // ----------------------------------------------------------------
    // Load server certificate (DER) and private key (BCrypt blob)
    // ----------------------------------------------------------------
    u32 cert_len, key_len;

    byte* cert_der = load_file("server.cer\0", @cert_len);
    if (cert_der == (byte*)0)
    {
        print("Failed to load server.cer\n\0");
        cleanup();
        return 1;
    };

    byte* key_blob = load_file("server.key\0", @key_len);
    if (key_blob == (byte*)0)
    {
        print("Failed to load server.key\n\0");
        ffree((u64)cert_der);
        cleanup();
        return 1;
    };

    print("Certificate and key loaded\n\0");

    // ----------------------------------------------------------------
    // Import the private key into BCrypt
    // The key file is expected to be a BCRYPT_RSAPRIVATEKEYBLOB.
    // Swap to import_private_blob from ecdsa:: if using an EC key.
    // ----------------------------------------------------------------
    RsaKey server_key;

    if (!rsa::import_private_blob(@server_key, key_blob, key_len))
    {
        print("Failed to import private key\n\0");
        ffree((u64)cert_der);
        ffree((u64)key_blob);
        cleanup();
        return 1;
    };

    ffree((u64)key_blob);
    print("Private key imported\n\0");

    // ----------------------------------------------------------------
    // Load the certificate into a Crypt32 context
    // ----------------------------------------------------------------
    Certificate server_cert;

    if (!x509::load_der(@server_cert, cert_der, cert_len))
    {
        print("Failed to load certificate\n\0");
        ffree((u64)cert_der);
        server_key.__exit();
        cleanup();
        return 1;
    };

    ffree((u64)cert_der);
    print("Certificate loaded\n\0");

    // ----------------------------------------------------------------
    // Link the BCrypt private key to the certificate context so
    // Schannel can use it during the TLS handshake
    // ----------------------------------------------------------------
    if (!x509::attach_ncrypt_key(@server_cert, @server_key.hNKey))
    {
        print("Failed to attach private key to certificate\n\0");
        server_cert.__exit();
        server_key.__exit();
        cleanup();
        return 1;
    };

    print("Private key linked to certificate\n\0");

    // ----------------------------------------------------------------
    // Set up the listening socket
    // ----------------------------------------------------------------
    socket server_socket(socket_type.TCP);
    server_socket.fd = tcp_socket();

    if (!server_socket.is_open())
    {
        print("Failed to create socket\n\0");
        server_cert.__exit();
        server_key.__exit();
        cleanup();
        return 1;
    };

    if (!server_socket.bind((i16)8443))
    {
        print("Failed to bind to port 8443\n\0");
        server_socket.close();
        server_cert.__exit();
        server_key.__exit();
        cleanup();
        return 1;
    };

    if (!server_socket.listen(5))
    {
        print("Failed to listen on socket\n\0");
        server_socket.close();
        server_cert.__exit();
        server_key.__exit();
        cleanup();
        return 1;
    };

    print("Starting server on port 8443...\n\0");
    print("Server listening on port 8443\n\0");
    print("Waiting for connections...\n\0");

    // Accept one client connection (mirrors original single-client design)
    sockaddr_in client_addr;
    socket client_socket(socket_type.TCP);
    client_socket.fd = tcp_server_accept(server_socket.fd, @client_addr);
    client_socket.connected = true;
    client_socket.remote_addr = client_addr;

    if (!client_socket.is_open())
    {
        print("Failed to accept client connection\n\0");
        server_socket.close();
        server_cert.__exit();
        server_key.__exit();
        cleanup();
        return 1;
    };

    print("Client connected from \0");
    print(client_socket.get_remote_ip());
    print(":\0");
    print((int)client_socket.get_remote_port());
    print("\n\0");

    // ----------------------------------------------------------------
    // Wrap the accepted socket in a TLS connection
    // ----------------------------------------------------------------
    TlsConn tls(client_socket.fd);

    if (!conn::server_create_cred(@tls, server_cert.ctx))
    {
        print("Failed to acquire server TLS credentials\n\0");
        client_socket.close();
        server_socket.close();
        server_cert.__exit();
        server_key.__exit();
        cleanup();
        return 1;
    };

    print("Performing TLS handshake...\n\0");

    if (!conn::server_handshake(@tls))
    {
        print("TLS handshake failed (error \0");
        print(tls.error_state);
        print(")\n\0");
        tls.__exit();
        client_socket.close();
        server_socket.close();
        server_cert.__exit();
        server_key.__exit();
        cleanup();
        return 1;
    };

    print("TLS handshake complete!\n\0");

    // ----------------------------------------------------------------
    // Echo loop — identical logic to original, now through TLS
    // ----------------------------------------------------------------
    byte[1024] buffer;

    while (true)
    {
        int bytes_received = conn::tls_recv(@tls, buffer, 1024u);

        if (bytes_received <= 0)
        {
            print("Client disconnected\n\0");
            break;
        };

        buffer[bytes_received] = '\0';

        print("Received (\0");
        print(bytes_received);
        print(" bytes): \0");
        print(buffer);
        print("\n\0");

        int bytes_sent = conn::tls_send(@tls, buffer, (u32)bytes_received);

        if (bytes_sent < 0)
        {
            print("Failed to send data to client\n\0");
            break;
        };

        print("Echoed back \0");
        print(bytes_sent);
        print(" bytes\n\0");
    };

    // ----------------------------------------------------------------
    // Clean up
    // ----------------------------------------------------------------
    conn::tls_shutdown(@tls);
    tls.__exit();

    client_socket.close();
    server_socket.close();
    server_cert.__exit();
    server_key.__exit();
    cleanup();

    print("Server shut down\n\0");
    return 0;
};
