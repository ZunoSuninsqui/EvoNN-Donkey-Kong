import subprocess

if __name__ == "__main__":
    # Entrena 50 generaciones
    subprocess.run(["python", "train_neat_dk.py", "--generations", "50"])
