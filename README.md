# Plasticity

To run code for `random_mnist.py`
`nohup python random_mnist.py --num-tasks=50 --save > random512.log 2>&1 &`

To run code for `soft_label_mnist.py`
`nohup python soft_label_mnist.py --num-tasks=50 --num-epochs=1000 --num-per-class=512 --continuous --k=3 --save > continuous.log 2>&1 &`
