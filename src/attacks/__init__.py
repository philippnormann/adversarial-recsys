from enum import Enum


class AttackMethod(Enum):
    FGSM = 'fgsm'
    PGD = 'pgd'
    CW = 'cw'

    def __str__(self):
        return self.value
