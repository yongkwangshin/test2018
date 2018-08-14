import argparse
import glob
import os


parser = argparse.ArgumentParser()
parser.add_argument("--train_images", type = str  )
parser.add_argument("--train_annotations", type = str  )


args = parser.parse_args()


images_path = args.train_images
segs_path = args.train_annotations



img = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
img.sort()	
seg  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
seg.sort()
print('first seg',seg)

n=len(img)
m=len(seg)
print('n',n,'m',m)


img_name=[]
seg_name=[]
count=[]

for i in range(n):
    for j in range(m):
        img_path=img[i]
        seg_path=seg[j]
        print(i,j)
        print(img_path,seg_path)
        img_name=os.path.splitext(os.path.basename(img_path))[0]
        seg_name=os.path.splitext(os.path.basename(seg_path))[0]
        print(img_name,seg_name)
        if img_name == seg_name:
            count.append(j)
            print('ddd')
            break
print(seg)
seg2=[]
for j in count:
    seg2.append(seg[j])

print('seg2',seg2)

print(os.path.basename(seg2[0]))

new_path='data3/'


pic_path='data2/'
os.chdir(pic_path)


for i in range(len(seg2)):
    seg2_path=seg2[i]
    seg2_pre_name=os.path.basename(seg2_path)
    seg2_name=os.path.splitext( seg2_pre_name )[0]
    print(seg2_name,new_path,'test')
    new_path_and_name = new_path + seg2_pre_name
    os.renames( seg2_pre_name, new_path_and_name)


n=len(img)
m=len(seg)
print('n',n,'m',m)



"""
os.path.basename(seg2)



os.mkdir("data3/")
for i in range(seg2):
    new_path='data3/'+ os.path.basename(seg2)[i]
    os.renames( os.path.basename(seg2)[i], new_path  )


실행명령
python  test.py --train_images="data1/" --train_annotations="data2/"

이름만 추출하기
path = "a/b/c/abc.txt"
result = os.path.splitext(os.path.basename(path))[0]
print(result)

탭 에러 해결책
TabError: inconsistent use of tabs and spaces in indentation
-> use only tab or space


"""