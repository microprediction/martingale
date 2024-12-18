import secrets
import string

def str_hash(length=8):
    """
    Generate a secure random hash string.

    Args:
        length (int): The length of the hash string to generate.

    Returns:
        str: A random hash string consisting of letters and digits.
    """
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))

# Example usage:
if __name__ == "__main__":
    hash_str = str_hash(8)
    print(f"Generated Hash: {hash_str}")  # e.g., 'G5kd92Ls'
