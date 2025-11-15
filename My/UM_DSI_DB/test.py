A = [ [123,4], [56,78], [90,12] ] 
B = [ [34,5], [67,89], [10,11] ]
for ap, rssi in A:
    print(f'AP: {ap}, RSSI: {rssi}')

row_data_list = []

row_data1 = {int(ap): int(rssi) for ap, rssi in A}
row_data2 = {int(ap): int(rssi) for ap, rssi in B}
row_data_list.append(row_data1)
row_data_list.append(row_data2)
print(row_data_list)