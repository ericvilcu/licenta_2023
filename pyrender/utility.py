from typing import Iterable,Generic,TypeVar

_T = TypeVar('_T')
class spin_iter(Generic[_T]):
    def __init__(self,i:Iterable[_T]) -> None:
        self.col=i
        self.iter=iter(self.col)
        
    def next(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter=iter(self.col)
            return next(self.iter)
        
    def __iter__(self):
        return self
        
    def __next__(self):
        return self.next()
    
    def __len__(self):
        return len(self.col)
