"""
This code is to generator vocab.txt file from vector.txt.
Format:
the 50
is 50
"""
def vo_gen(ifilename,ofilename):
	with open(ifilename, "r") as ins:
		content = []
		for line in ins:
			unit = line.split()
			print unit[0]
			


vo_gen('glove.6B.300d.txt','vocab.txt')

