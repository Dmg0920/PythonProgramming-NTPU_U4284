
from typing import Any,Union,Optional,Type,NewType
#----------------------------------------------------------------
                     # Complex type
#----------------------------------------------------------------
def h1(x: list[int]) -> float:
    '''Harmonic Mean'''
    return len(x) / sum([1/k for k in x])
# Example:
# h1([1, 2, 4])
def h2(x: list[int]) -> Optional[float]:  # 改成 Optional[float]
    '''Harmonic Mean'''
    return len(x) / sum([1/k for k in x])

# Example:
# h2([1, 2, 4])

def h3(x: list[int]) -> Union[float, int]:  # 改成 Union[float, int]
    '''Harmonic Mean'''
    return len(x) / sum([1/k for k in x])

# Example:
# h3([1, 2, 4])

def h4(x: list[int]) -> float:  # 改成 float，不用 Any
    '''Harmonic Mean'''
    return len(x) / sum([1/k for k in x])

# Example:
# h4([1, 2, 4])

#----------------------------------------------------------------
                     # User-defined Type
#----------------------------------------------------------------

class Student:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

# Example:
# Student("Amy", 20)

# Student is a user-defined type, 
# and it's used as a type hint in the print_student_details function
def print_student_details(student: Student) -> None:
    print(student.name, student.age)

# Example:
# print_student_details(Student("Amy", 20))

# create_student expects a Student class (or a subclass of Student) 
# as its first argument.
def create_student(cls: Type[Student], name: str, age: int) -> Student:
    return cls(name, age)

# Example:
# create_student(Student, "Bob", 21)

# you want to make sure you don't mix them up. 
# Both are represented as integers, 
# so you can use NewType to create two distinct types:
StudentID = NewType('StudentID', int)
CourseID = NewType('CourseID', int)
def get_student(student_id: StudentID) -> None:
    pass

# Example:
# get_student(StudentID(1001))
def enroll_in_course(student_id: StudentID, course_id: CourseID) -> None:
    pass

# Example:
# enroll_in_course(StudentID(1001), CourseID(501))
