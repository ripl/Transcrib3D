def get_principle(utterance,use_priority=False):
    prompt=''
    prompt=prompt+"""Tips: while multiple objs may appear within the description, it points to only 1 focal object, with the other objects serving to aid in locating or contextualizing it. For instance, spatial relation with other objects might be employed to establish the position or orientation of this focal object. Examples:
    1.'The brown cabinet covers the entire back wall. There is a door with a blue sign located between the brown cabinet.' The first sentence is actually a noun phrase starting with 'the,' indicating that the focal object being described is 'the brown cabinet.' The second sentence describes the spatial relationship between the door and the brown cabinet, providing supplementary details about the described brown cabinet.
    2.'This is a big exercise ball. The ball is under the table.' The first sentence starts with 'this is,' indicating the object being described, which is a 'big exercise ball.' The second sentence is used to provide additional information about the ball's location.
    3.'In the corner of the kitchen, there are three trash cans. Beside the third trash can from the left, there's a white stool.' The description first sets up a scene with three trash cans, and then move on to describe the location of the white stool in relation to the trash cans. Therefore, the white stool is the target object."""
    if use_priority:
        prompt=prompt+"\nConsider different constraints in order (1 to 7) & priority (1 highest, 7 lowest):"
        prompt=prompt+"\n1: Obj name(category). Names in description & obj list may differ (e.g. similar names such as 'table' and 'desk', 'trash can' and 'recycling bin', 'coffee table' and 'end table'), so use common sense to find all possible candidate objects, ensure no missing, don't write code. If only 1 object in list has the same/similar category with the one described object, answer it directly, discard other constraints. For instance, with description 'the black bag left to the couch' and only 1 bag in the scene, answer it directly, discard 'black' and 'left' constrains."
        prompt=prompt+"\n2: Horizontal relation like 'next to''farthest''closest''nearest''between''in the middle''at the center'(if given)(not include 'behind''in front of'). Consider only center x,y,z coords of objs, disregard sizes."
        prompt=prompt+"\n3: Color(if given). Be lenient with color, RGB values in obj list & standard RGB value of obj in description may differ significantly. You can use distance in RGB color space as a metric."
        prompt=prompt+"\n4: Size & shape(if given). Be cautious not to make overly absolute judgments about obj size. E.g., 'a tiny trash can' doesn't necessarily refer to smallest one in terms of volume."
        prompt=prompt+"\n5: Direction relation 'left''right'(if given). To judge A on 'left' or 'right' of B, calc vec observer-A & observer-B(both projected to x-y plane). If cross product of vec observer-A & vector observer-B(in this order) has positive z, A on right of B. If z neg, A on left of B. Note that order of cross product matters, put vec observer-A at first. Consider which two objs' left-right relation needs to be determined in sentence, that is, which is A & which is B. DON'T determine left & right relation by compare x or y coords." 
        prompt=prompt+"\n6: Direction relation 'in front of' and 'behind'(if given). Use 'spatially closer' to replace them. To determine which object, P1 or P2, is behind Q, calculate their distances from Q. The one with the smaller distance is behind Q. It is the same for judging 'in front of': also smaller distance. DON'T determine front & behind relation by compare x or y coords." 
        prompt=prompt+"\n7: Vertical relation like 'above'and'under''on''sits on'(if given). Consider only center coords of objs, disregard sizes. Be more lenient with this."
        prompt=prompt+"\nExplicitly go through these 7 constraints. For every constraint, if it is not mentioned in description, tell me and skip; if mentioned, apply this constraint and record the results of each candidates. For constraint 1, use common sense, no code. For others, write code, which should print the metrics of each candidate objects, instead of only print the most possible object id. After going through all constriants, evaluate all results comprehensively basing on 1-7 priority, and choose the unique target object."
    else:
        prompt=prompt+"""While multiple objs may appear within the description, it points to only 1 focal object, with the other objects serving to aid in locating or contextualizing it. For instance, spatial relation with other objects might be employed to establish the position or orientation of this focal object. So first you should identify this focal object(that is, the category name of it) from the description.
        Next, you can identify potential objects from the object list based on the category name of the focal object. You should rely on your common sense to comprehensively identify all relevant candidates without writing code. For example, for the category name 'table,' objects such as 'table,' 'desk,' 'end table,' 'coffee table,' and so on from the object list should all be considered as potential candidates.
        Then, count(do not write code) and tell me the number of candidate objects. If it is 1, which means only one candidate object, you must directly choose it as answer, then stop your response. For example, if the description is 'the white bathhub on the left of the toilet' and there is only one 'bathhub'-like object in the list, answer it directly, ignore 'white' and 'left of the toilet' constraints.
        If there are multiple candidate objects, you can continue. Identify the constraints in the description. There might be multiple constraints to help finding the unique target object from multiple candidate objects. For each constraint, you can define a quantitative metric to assess the degree to which each candidate object satisfies this constraint. 
        You can write code to calculate the metrics, printing out the metrics of each candidate objects, instead of only print the most possible object id.
        Some special tips for some constraints:
        - Color(if given). Be lenient with color, because different shades of color mentioned in description can have different RGB values
        - Direction relation 'left''right'(if given). To judge obj A on 'left' or 'right' of B, calc vector observer-A & observer-B(both projected to x-y plane). If cross product of vector observer-A & vector observer-B(in this order) has positive z, A on right of B. If z neg, A on left of B. Note that order of cross product matters, put vec observer-A at first. DON'T determine left & right relation by comparing x or y coords.
        - Direction relation 'in front of' and 'behind'(if given). Use 'spatially closer' to replace them. To determine which object, P1 or P2, is behind Q, calculate their distances from Q. The one with the smaller distance is behind Q. It is the same for judging 'in front of': also smaller distance. DON'T determine front & behind relation by comparing x or y coords.
        - Vertical relation such as 'on''above''under'(if given). If obj M has vertical relation with obj N, the x,y coord of ctr of M should be inside the x,y range of obj N, while z of M and z of N should satisfy the corresponding order.
        After going through all constraints in the description, double check the description:'%s' , and evaluate all results and metrics comprehensively, then choose the unique target object."""%utterance

    prompt=prompt+"\nPerceive wall as plane. Distance from obj to wall=vert dist to plane, not to wall center. Wall front=side of plane where obj exist."

    return prompt

def get_principle_sr3d():
    prompt=""
    # prompt=prompt+"\nYou must comprehensively consider the x, y, z coordinates, not only one of them (for example, you should consider both greater z and similar x, y coordinates when judging vertical relation). "
    # prompt=prompt+"\nWhen determining vertical relation such as 'above''under''on''on top of''support'(if given). If obj M has vertical relation with obj N, the x,y coord of ctr of M should be inside the x,y range of obj N, while z of M and z of N should satisfy the corresponding order. You can igonre the size of objects and only consider ctr coords here."
    prompt=prompt+"\nWhen determining vertical relation such as 'above''under''on''on top of''support'(if given). For example, if obj M is on top of / supportted by obj N, the x,y coord of ctr of M should be inside the x,y range of obj N, while z of M is greater than z of N. You can igonre the size in z direction of objects here. If  you cannot find the obj M after several tries, you can choose one which is closest to N."
    prompt=prompt+"\nWhen determining the orientation of object B relative to object A, you should calculate the angle between the x-y plane vector from A to B(projected onto x-y plane) and one of the direction vectors of A (the one that corresponds to the direction mentioned in the problem). The smaller the angle, the more it indicates that B is in the corresponding direction of A." 
    return prompt

def get_system_message():
    system_message="Imagine you are an artificial intelligence assitant with a python interpreter. So when answering questions, you can choose to generate python code (for example, when there is need to do quantitative evaluation). The generated code should always use print() function to print out the result and keep two decimal places for numbers. The code should be written in python, start with '```python\nimport numpy as np\nimport math\n' and end with '```'. Keep your code and comments concise. When answer step by step, stop whenever you feel there is need to generate python code (for example, where there is need to do quantitative evaluation) and wait for the result from the code execution. Make sure your code will print out something(include failure info like 'nothing found'), especially when you use if logic." # Before generating code, say 'Let's write some python code to get the results.', then stop. You'll receive an empty message from user, then you start to generate code. If you are printing thing like 'metric: value', make it clear what the metric is."

    return system_message