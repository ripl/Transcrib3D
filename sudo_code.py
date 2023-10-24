
def detect_object():
    # detect object in the scene and return their 3d bounding boxes, colors and labels]
    # cat_list=[]
    # img,depth_img=get_img()
    # for cat in cat_list:
    #     mask=mdetr(img,str(cat))
    #     info=
    pass

def find_object(description:str, objects_info):
    # use current GPT-refer model to find the unique object in objects based on description
    # return the boudning box of the object
    pass

def grab(grab_point,drop_point):
    # action: grab object from point start to point end
    pass

def split_instruction(instruction:str):
    pass

def put_A_on_B(A:str, B:str):
    objects_info=detect_object()
    A_box=find_object(description=A, objects_info=objects_info)
    B_box=find_object(description=B, objects_info=objects_info)

    grab_point=A_box['center_position']
    drop_point=B_box['center_position']
    drop_point[2]+=(A_box['size'][2]/2+B_box['size'][2]/2)

    grab(grab_point=grab_point, drop_point=drop_point)

instruction="Put the duck into the pink cup furthest from the cube"
text_A,text_B=split_instruction(instruction)

# GPT generate the code:
# put_A_on_B(A="the duck",B="the pink cup furthest from the cube")
put_A_on_B(A=text_A, B=text_B)