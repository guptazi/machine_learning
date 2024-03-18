import math
with open('top10-worst.txt','r') as read_obj:
    csv_file=read_obj.readlines()

distance_from_tsu=[]
tsu_lat=36.16963449238665
tsu_lon=-86.82562299320742



def calculate_distance(lat1, lon1, lat2, lon2):
    lat1=57.296*lat1
    lon1=57.296*lon1
    lat2=57.296*lat2
    lon2=57.296*lon2
    distance = math.acos(math.sin(lat1) * math.sin(lat2) +
                         math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)) * 6371
    return distance

for data in csv_file[1:]:
    parts=data.strip().split(',')
    lat=float(parts[2])
    lon=float(parts[3])
    mag=float(parts[5])
    distance=calculate_distance(lat,lon,tsu_lat,tsu_lon)
    distance_from_tsu.append((data[0], distance))

with open('dist-to-TSU.txt', 'w') as file:
    file.write("Earthquake Data,Distance to TSU (km)\n")
    for earthquake, distance in distance_from_tsu:
        file.write(f"{earthquake.strip()},{distance}\n")
