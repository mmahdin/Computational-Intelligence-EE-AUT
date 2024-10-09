# Define a max header size of 64 characters
MAX_HEADER_SIZE = 2

# Example of protocol headers for each OSI layer
HEADERS = {
    'application': 'APP_HDR_',
    'presentation': 'PRESENT_HDR_',
    'session': 'SESSION_HDR_',
    'transport': 'TRANSPORT_HDR_',
    'network': 'NETWORK_HDR_',
    'data_link': 'DATALINK_HDR_',
    'physical': 'PHYSICAL_HDR_'
}

# Layer 7: Application layer


def application_layer(message, size):
    header = HEADERS['application'].ljust(MAX_HEADER_SIZE)
    new_message = header + message
    print(f"Application Layer: {new_message}")
    presentation_layer(new_message, len(new_message))

# Layer 6: Presentation layer


def presentation_layer(message, size):
    header = HEADERS['presentation'].ljust(MAX_HEADER_SIZE)
    new_message = header + message
    print(f"Presentation Layer: {new_message}")
    session_layer(new_message, len(new_message))

# Layer 5: Session layer


def session_layer(message, size):
    header = HEADERS['session'].ljust(MAX_HEADER_SIZE)
    new_message = header + message
    print(f"Session Layer: {new_message}")
    transport_layer(new_message, len(new_message))

# Layer 4: Transport layer


def transport_layer(message, size):
    header = HEADERS['transport'].ljust(MAX_HEADER_SIZE)
    new_message = header + message
    print(f"Transport Layer: {new_message}")
    network_layer(new_message, len(new_message))

# Layer 3: Network layer


def network_layer(message, size):
    header = HEADERS['network'].ljust(MAX_HEADER_SIZE)
    new_message = header + message
    print(f"Network Layer: {new_message}")
    data_link_layer(new_message, len(new_message))

# Layer 2: Data Link layer


def data_link_layer(message, size):
    header = HEADERS['data_link'].ljust(MAX_HEADER_SIZE)
    new_message = header + message
    print(f"Data Link Layer: {new_message}")
    physical_layer(new_message, len(new_message))

# Layer 1: Physical layer


def physical_layer(message, size):
    header = HEADERS['physical'].ljust(MAX_HEADER_SIZE)
    new_message = header + message
    print(f"Physical Layer: {new_message}")
    # No lower layer after this one

# Main function to initiate message flow


def main():
    application_message = input("Enter the application message: ")
    size = len(application_message)
    application_layer(application_message, size)


if __name__ == "__main__":
    main()
