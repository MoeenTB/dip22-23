import numpy as np

n_list1 = (54000-10) * np.random.random(80) + 10

print("\nTask two:")
# print(type(n_list1), type(n_list1[0]), sep = '\n')
print(type(n_list1), n_list1.dtype, sep = '\n')

n_list2 = np.round(n_list1)

n_list3 = n_list2.astype(np.uint16)
print("\nTask four:")
print(n_list3.dtype)

n_list4 = np.reshape(n_list3, (8, 10))
print("\nTask five:")
print(n_list4)

print("\nTask six:")
print(f'Minimum: {np.amin(n_list4)} \t Maximum: {np.amax(n_list4)}')

n_list5 = n_list4.astype(np.int8)
print("\nTask seven:")
print(n_list5)  # uint8 is in [0, 255] and therefore will change the values of our array.

C_two = tuple(n_list5[:, 1]) # or to_tuple
R_three = list(n_list5[2, 1:])
print("\nTask eight:")
print("C_two: ", C_two)
print("R_three: ", R_three)

n_dict = dict(zip(C_two, R_three))
print("\nTask nine:")
print(n_dict)
