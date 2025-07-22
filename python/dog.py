class Dog:
    
    def __init__(self, name, age):
        self.name = name
        self.species = None
        self.age = age

    def descriptor(self):
        return f"{self.name} is {self.age} year old you idios"

    def speak(self, sound):
        return f"{self.name} says {sound}"


class JackRussellTerrier(Dog):

    pass

class Dachshund(Dog):
    pass

class Bulldog(Dog):
    pass

miles = JackRussellTerrier("Miles", 4)
a = Dog("Something", 4)
b = Dog("love", 3)

print(miles.speak("le rat"))