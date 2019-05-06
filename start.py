import load_data as ld
import network as nt

training_data, validation_data, test_data = ld.load_wrapper_data ()
net = nt.Network ([784, 100, 10])
net.SGD (training_data, 30, 10, 3.0, test_data=test_data)