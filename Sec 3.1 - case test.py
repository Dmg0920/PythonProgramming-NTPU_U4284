
from typing import Any,Union,Optional,Type,NewType
#----------------------------------------------------------------
                     # Complex type
#----------------------------------------------------------------
def h1(x: list[int]) -> float:
    '''Harmonic Mean'''
    return len(x) / sum([1/k for k in x])
# Using Optional
def h2(x: list[int]) -> Optional[str]:
    '''Harmonic Mean'''
    return len(x) / sum([1/k for k in x])
# Using Union
def h3(x: list[int]) -> Union[str,int]:
    '''Harmonic Mean'''
    return len(x) / sum([1/k for k in x])
# Using Any
def h4(x: list[int]) -> Any:
    '''Harmonic Mean'''
    return len(x) / sum([1/k for k in x])

#----------------------------------------------------------------
                     # User-defined Type
#----------------------------------------------------------------

class Student:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

# Student is a user-defined type, 
# and it's used as a type hint in the print_student_details function
def print_student_details(student: Student) -> None:
    print(student.name, student.age)

# create_student expects a Student class (or a subclass of Student) 
# as its first argument.
def create_student(cls: Type[Student], name: str, age: int) -> Student:
    return cls(name, age)

# you want to make sure you don't mix them up. 
# Both are represented as integers, 
# so you can use NewType to create two distinct types:
StudentID = NewType('StudentID', int)
CourseID = NewType('CourseID', int)
def get_student(student_id: StudentID) -> None:
    pass
def enroll_in_course(student_id: StudentID, course_id: CourseID) -> None:
    pass