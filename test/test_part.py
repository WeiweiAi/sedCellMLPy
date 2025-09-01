from functools import partial

def send_email(server, port, username, password, recipient, message):
    print(f"Connecting to {server}:{port} as {username}")
    print(f"Sending message to {recipient}: {message}")

# Pre-fill server, port, username, and password
send_from_default_account = partial(
    send_email,
    server="smtp.example.com",
    port=587,
    username="user@example.com",
    password="securepassword"
)

# Now only recipient and message are needed
send_from_default_account(recipient="alice@example.com", message="Hello Alice!")

send_from_default_account = partial(
    send_email,
    server="smtp.example.com",
    port=300,
    username="user@example.com",
    password="securepassword"
)

send_from_default_account(recipient="alice@example.com", message="Hello Alice!")