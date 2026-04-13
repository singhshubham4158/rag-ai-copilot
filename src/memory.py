class Memory:
    def __init__(self):
        self.history = []

    def add(self, q, a):
        self.history.append((q, a))

    def get(self):
        return "\n".join([f"Q:{q} A:{a}" for q,a in self.history[-5:]])