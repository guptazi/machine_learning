#qestion 1

with open('Homework3Data.csv','r') as read_obj:
    csv_file=read_obj.readlines()

earthquake_list=[]
print(csv_file)

for data in csv_file[1:]:
    parts=data.strip().split(',')
    magnitude=float(parts[5])
    earthquake_list.append((data,magnitude))

earthquake_list.sort(key=lambda x: x[1], reverse=True)
top_10 = earthquake_list[:10]

with open('top10-worst.txt','w') as file:
    file.write(csv_file[0])
    for earthquake in top_10:
        file.write(earthquake[0])
