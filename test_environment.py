# Python
def test_environment():
    try:
        print("Environment setup is working correctly!")
    except Exception as e:
        print(f"Something went wrong: {e}")

if __name__ == "__main__":
    test_environment()