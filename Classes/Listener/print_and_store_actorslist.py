from Classes.Listener.listener_actor_storing import ListenerActorStoring
from Classes.Listener.listener_actor_printing import  ListenerActorPrinting
from Library.Listener.listener_actor_list import ListenerActorList
'''
class PrintAndStoreActorsList():
    def __call__(self):
        return ListenerActorList([
            ListenerActorPrinting(),
            ListenerActorStoring()
        ])
'''

class PrintAndStoreActorsList(ListenerActorList):
    def __init__(self):
        super().__init__([
            ListenerActorPrinting(),
            ListenerActorStoring()
        ])
