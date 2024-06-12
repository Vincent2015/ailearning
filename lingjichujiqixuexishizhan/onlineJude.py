while True :
	try :
			line = input()
			a = line.split()
			print(int(a[0])+int(a[1]))
	except:
			print('error')
			break
	
