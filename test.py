class A:
    def say(utterance):
        print(utterance)
    def get_name(self,name):
        self.name=name
        self.say("My name is:")
        self.say(name)

a=A()
a.get_name("BOB")