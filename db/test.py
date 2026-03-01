def main():
    inp=input()
    l=int(inp.split()[0])
    h=int(inp.split()[1])

    c=0 #possible combinations
    for i in range(l,h):
        if validate_int(i):
            c+=1

    print(c)

def validate_int(num: int):
    original_num=num
    arr=[]
    j=0
    while num!=0:
        dig=num%10
        if dig!=0 and dig not in arr and original_num%dig==0:
            arr.append(dig)
        else:
            return False
        num//=10
        j+=1
    return True

if __name__=="__main__":
    main()