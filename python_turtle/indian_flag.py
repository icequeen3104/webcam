import turtle
flag = turtle.Turtle()
flag.speed(1)
flag.begin_fill()
flag.fillcolor("orange")
flag.fd(100)
flag.left(90)
flag.forward(30)
flag.left(90)
flag.forward(100)
flag.left(90)
flag.fd(30)
flag.end_fill()
flag.fd(30)
flag.left(90)
flag.fd(100)
flag.left(90)
flag.fd(30)
flag.fd(30)
flag.left(90)
flag.fd(100)
flag.left(90)
flag.fd(60)
flag.fd(30)
flag.left(90)
flag.fd(100)
flag.left(90)
flag.fd(90)
flag.left(90)
flag.fd(100)
flag.left(90)
flag.fd(300)
flag.bk(300)
flag.forward(60)
flag.left(90)
flag.fd(50)
#flag.begin_fill()
#flag.fillcolor("blue")
for i in range(0,24):
    flag.fd(15)
    flag.left(165)
    flag.fd(15)
    flag.left(180)

flag.right(90)
flag.circle(15)
flag.end_fill()
flag.bk(50)
flag.begin_fill()
flag.fillcolor("green")
flag.fd(100)
flag.right(90)
flag.fd(30)
flag.right(90)
flag.fd(100)
flag.end_fill()
flag.hideturtle()




