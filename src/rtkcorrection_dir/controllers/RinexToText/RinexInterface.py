from abc import ABC, abstractmethod

class RinexInterface(ABC):
    """
    the idea of interface is design the class but without implementation
    # this decrator force the user to use this method in the subclass
    """
    def __init__(self,filename: str):
        self.filename = filename

    @abstractmethod
    def GPS(self):
        pass

    @abstractmethod
    def Galileo(self):
        pass

    @abstractmethod
    def GLONASS(self):
        pass

    @abstractmethod
    def BeiDou(self):
        pass
    
    @abstractmethod
    def NavIC(self):
        pass

    @abstractmethod
    def QZSS(self):
        pass