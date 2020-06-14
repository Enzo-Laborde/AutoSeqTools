# -*- coding: utf-8 -*-

import sys



########### ORIGINAL SCRIPT FOR PPL ######

def palindromicLengths(w):
	"""
	Computes the palindromic lengths of the prefixes of the input word.
	"""

	"""
	This code uses Eertree for speed, but does not implement the most efficient
	version described in Section 4 of [2].
	"""

	T = Eertree()
	n = len(w)
	ans = [0] + [n] * n
	d = [0] * (n)

	for i, c in enumerate(w):
		T.add(c)

		v = T.maxSufT
		while v.len > 0:
			ans[i + 1] = min(ans[i + 1], ans[i + 1 - v.len] + 1)
			d[i] = ans[i+1] - ans[i] # string for variation of palindromic lengths
			v = v.link

	return ans[1:], d

class EertreeNode:

	def __init__(self):
		self.edges = {} # edges (or forward links)
		self.link = None # suffix link (backward links)
		self.len = 0 # the length of the node

class Eertree:
	"""
	Implements the online version of the Eertree building algorithm.
	"""

	"""
	Implementation from https://rosettacode.org/wiki/Eertree
	"""

	def __init__(self, w = None):
		self.nodes = []
		# The two root nodes.
		self.rto = EertreeNode() # node -1 for odd length palindromes
		self.rte = EertreeNode() # node 0 for even length palindromes

		# Initialize empty tree
		self.rto.link = self.rto
		self.rte.link = self.rto
		self.rto.len = -1
		self.rte.len = 0
		self.S = [0] # The string accumulated so far, T = S[1..i]
		self.maxSufT = self.rte # Maximum suffix of tree T.

		if w is not None and len(w) > 0:
			for c in w:
				self.add(c)

	def get_max_suffix_pal(self, startNode, a):
		"""
		Find longest palindromic suffix that is preceded by the letter a. If it
		does not exist, return the -1 node.
		"""

		# Traverse the palindromic suffixes of T in the order of decreasing length.
		# For each such palindrome, we check if T[i - k] is a or we eventually
		# arrive at the -1 node.
		u = startNode
		i = len(self.S)
		k = u.len
		while u is not self.rto and self.S[i - k - 1] != a:
			u = u.link
			k = u.len

		return u

	def add(self, a):
		"""
		Add a new letter into the Eertree structure.
		"""

		# Find the longest palindromic suffix of T preceded by the letter a.
		Q = self.get_max_suffix_pal(self.maxSufT, a)

		# Check if Q has an outgoing edge labeled by a.
		createANewNode = not a in Q.edges

		if createANewNode:
			# We create the node P of length Q+2.
			P = EertreeNode()
			self.nodes.append(P)
			P.len = Q.len + 2
			if P.len == 1:
				# If P = a, create the suffix link to node 0.
				P.link = self.rte
			else:
				# It remains to create the suffix link from P if |P| > 1. Just continue
				# traversing the palindromic suffixes of T starting with the suffix
				# link of Q.
				P.link = self.get_max_suffix_pal(Q.link, a).edges[a]

			# Create the edge.
			Q.edges[a] = P

		# P becomes the new maxSufT.
		self.maxSufT = Q.edges[a]

		# Store accumulated input string.
		self.S.append(a)

		return createANewNode



########### ADDED SCRIPT #################
##			Made by Laborde Enzo 		##
##########################################

def dToStr(d):
	"""
	Transform the list d (=[1,-1,0,...]) to a string '+-0...'
	"""

	s = ''
	
	if type(d) == list:
		for i in range(len(d)):
			if '0' in str(d[i]):
				s += '0'
			elif '-' in str(d[i]):
				s += '-'
			else:
				s += '+'
		return s

	else:
		return d

def arguments(argv):
	"""
	Read all usefull arguments for the program. 
	"""

	# Error if they are missing arguments, or if we ask manual.
	if len(argv) < 1 or argv[0] == 'man' or argv[0].find('-') == 0:
		Help = open('man.txt', 'r')
		print(Help.read())
		if len(argv) == 0: print('Error: Not enough arguents\n')
		elif argv[0].find('-') == 0: print('Error: Morphism must be the first argument, and coding have to be in second if needed!\n')
		sys.exit()
	
	# Create the morphism and the coding :
	i = 97  # ASCII code for letters, to build the morphism (a-z, 0-9, A-Z)
	d = {}  # Morphism dictionary
	c = {}  # Coding dictionary
	make_coding = bool(argv[1].find('-'))  # Is False if second argument is not a coding but a parameter (start with '-' => False)
	
	# While the first argument don't have '-' in it (it's the morphism), we build the morphism dico d, and the coding dico c if asked, and stop when the new first argument is an parameter
	while argv[0].find('-') != 0:

		# Error if they are to many characters in the alphabet
		if i == 91:
			sys.exit('Error: Alphabet can only contain 62 different characters! a-z, 0-9 and A-Z.\n')

		# Build the dico d and c that contain  morphism and coding respectively ...
		if argv[0].find(',') != -1:
			d[chr(i)] = argv[0][:argv[0].find(',')]  # Get the string before the ','
			argv[0] = argv[0][argv[0].find(',') + 1:]  # Reduce argv[0] length

			# if another ',' left and if we use coding function ...
			if argv[1].find(',') != -1 and make_coding:  
				c[chr(i)] = argv[1][:argv[1].find(',')]  # Get the string before the ','
				argv[1] = argv[1][argv[1].find(',') + 1:]  # Reduce argv[1] length
			
			# Else coding size and morphism size are not equal !
			elif make_coding:
				sys.exit('\nMorphism length is greater than coding length!\n')

		# Take the last map for morphism and coding ...
		else:
			d[chr(i)] = argv[0]  # Get the left string (they are no ',')

			# If we make coding and if there is still a ',' : coding size and morphism size are not equal !
			if argv[1].find(',') != -1 and make_coding:
				sys.exit('Morphism length is smaller than coding length!\n')

			#Else if morphism and coding have same size ...
			elif make_coding:
				c[chr(i)] = argv[1]

			del argv[0]
			if make_coding: del argv[0]  # Delete coding argument if necessary     
		
		i += 1

		# If the morphism alphabet have more than 26 letters, use digits 0-9.
		if i == 123:
			i = 48
		
		# If the morphism alphabet have more than 36 letters, use capitalized letters A-Z.
		if i == 58:
			i = 65
	
	# List of other arguments
	args = {}
	while argv:
		s = argv.pop(0)

		# Len of the infinite word to compute, default will be 2048.
		if '-n' == s[:2] and s[2:] != '':
			args['-n'] = int(s[2:])

		### Kernel parameters
		# Value <value> of the k-kernel to compute on the <sequence> sequence.
		elif '-k' == s[:2]:
			if s[2:] != '':
				args['-k'] = s[2:]
			else:
				args['-k'] = '2w'

		# Up to which iteration of i you want to compute the Kernel
		elif '-i' == s[:2] and s[2:] != '':
			args['-i'] = int(s[2:])

		# length of sequences in the Kernel
		elif '-s' == s[:2] and s[2:] != '':
			args['-s'] = int(s[2:])

		# Size of sequences in the kernel to show on screen
		elif '-u' == s[:2] and s[2:] != '':
			args['-u'] = int(s[2:])

		# Number of times to multiply s.lenght by k.value (increase kernel subseq. length)
		elif '-z' == s[:2] and s[2:] != '':
			args['-z'] = int(s[2:])

		# Specify which Kernel you want to see the content, default is ''
		elif '-IShow' == s[:6]:
			if s[6:] != '':
				args['-IShow'] = int(s[6:])
			else:
				args['-IShow'] = ''
		
		### Morphism detection parameters for d(n) sequences (see '-f' boolean argument below)
		# Number of steps to check if morphism is good
		elif '-FStep' == s[:6]:
			if s[6:] != '':
				args['-FStep'] = int(s[6:])
			else:
				args['-FStep'] = 50
		
		# Max word length taken by morphism
		elif '-FWord' == s[:6]:
			if s[6:] != '':
				args['-FWord'] = int(s[6:])
			else:
				args['-FWord'] = 4
		
		# Max time the morphism image length is greater than the pre-image
		elif '-FMap' == s[:5]:
			if s[5:] != '':
				args['-FMap'] = int(s[5:])
			else:
				args['-FMap'] = 10
		
		### Complexity parameters
		# Compute complexity for value iteration on the sequence.
		elif '-c' == s[:2]:
			if s[2:] != '':
				args['-c'] = s[2:]
			else:
				args['-c'] = '10w'
		
		# Compute complexity for -CStep diferent length
		elif '-CStep' == s[:6]:
			if s[6:] != '':
				args['-CStep'] = max(1,int(s[6:]))
			else:
				args['-CStep'] = 1
		
		### Other arguments
		else:
			# Boolean arguments. 
			for a in ['-f', '-IVerb', '-m', '-FVerb', '-FCode', '--args']:      
				if a in s[0:len(a)]:
					args[a] = True

			# Show prefix of sequences below
			for a in ['-w', '-p', '-d']:
				if a in s[:len(a)]:
					if s[len(a):] != '':
						args[a] = int(s[len(a):])
					else:
						args[a] = ''
				
	# If we ask for k-Kernel but not specify some parameters, and for finding automaticlly good prefix length.
	if '-k' in args:
		if '-i' not in args:
			args['-i'] = 6
		if '-s' not in args:
			args['-s'] = 32
		if '-z' not in args:
			args['-z'] = 0
		if '-n' not in args:
			args['-n'] = ( int(args['-k'][:len(args['-k']) - 1]) ** (args['-z'] + args['-i'])) * args['-s']
		else:
			print('  /!\\ When using \'-k\', don\'t use \'-n\' argument /!\\ \n')  # If we give a length but ask for kernel
		if '-u' not in args:
			args['-u'] = args['-s'] * int(args['-k'][:len(args['-k']) - 1]) ** args['-z']
		if '-IShow' in args and args['-IShow'] == '':
			args['-IShow'] = args['-i']

	# If we ask for Finding morphism, but don't specify some variables.
	if '-f' in args:
		if '-FStep' not in args:
			args['-FStep'] = 50
		if '-FWord' not in args:
			args['-FWord'] = 4
		if '-FMap' not in args:
			args['-FMap'] = 10

	# If we ask for complexity, but don't specify some variables.
	if '-c' in args:
		if '-CStep' not in args:
			args['-CStep'] = 1

	if '-n' not in args:
		args['-n'] = 2048
	
	# If some arguments have empty string, we set the max value.
	for a in ['-w', '-p', '-d']:
		if a in args and args[a] == '':
			args[a] = args['-n']
	
	return d, c, args


def gen_d(length):
	"""
	Generate the sequences d(n) of sequence using morphism found with the 'IFMorphism' function.
	"""

	### d(n) morphism of Sierpinsky word
	# m = {'a': 'abc', 'b': 'bdd', 'c': 'efg', 'd': 'ddd', 'e': 'edh', 'f': 'ddf', 'g': 'ifg', 'h': 'edj', 'i': 'abj', 'j': 'hdj'} 
	# c = {
	# 	'a': '++-+00+--',
	# 	'b': '+00000000',
	# 	'c': '+0000-+--',
	# 	'd': '000000000',
	# 	'e': '+00000+0-',
	# 	'f': '00000000-',
	# 	'g': '++-00-+--',
	# 	'h': '+0000000-',
	# 	'i': '++-+0000-',
	# 	'j': '+0-00000-' }
	
	### d(n) morphism of Paper-Folding word
	m = {'a': 'ab', 'b': 'cd', 'c': 'ef', 'd': 'gd', 'e': 'eb', 'f': 'ch', 'g': 'if', 'h': 'gh', 'i': 'ib'}
	c = {
		'a': '+0+0-+0+000-++0-+-0+00+00-0++-0+000+00+0-+000+000+00000-0++0-+0-',
		'b': '0++-0+00+00-0+00000+00+00-0++-0+0+-+0-0+000+000+0+-+0-000+000+0-',
		'c': '0++-0+00+00-0+00000+00+00-0++-0+000+00+0-+000+000+00000-0++0-+00',
		'd': '+00+-00+0000+0000-0+00+00-0++-0+0+-+0-0+000+000+0+-+0-000+000+0-',
		'e': '0++-0+00+00-0+00000+00+00-0++-0+000+00+0-+000+000+00000-0++0-+0-',
		'f': '0++-0+00+00-0+00000+00+00-0++-0+0+-+0-0+000+000+0+-+0-000+000+00',
		'g': '0+00000+00000+000-0+00+00-0++-0+000+00+0-+000+000+00000-0++0-+00',
		'h': '+00+-00+0000+0000-0+00+00-0++-0+0+-+0-0+000+000+0+-+0-000+000+00',
		'i': '0+00000+00000+000-0+00+00-0++-0+000+00+0-+000+000+00000-0++0-+0-' }

	### d(n) morphism of Rudin-Shapiro word
	# m = {'a': 'ab', 'b': 'cd', 'c': 'eb', 'd': 'ed', 'e': 'cb'} 
	# c = {
	# 	'a': '+00+00000-++00-++00-+0+00+00+00+-0+00-+00+-0+0+0-0+00-+00+00+00+',
	# 	'b': '0+0-0++-00+0+0-++00-+0+00+00+00+0+0-0++-00+0+0-+000+0-+00+00+00+',
	# 	'c': '-0+00-+00+-0+0+00+00-++-+00+000+0-+000+-+0-0+0+0-0+00-+00+00+00+',
	# 	'd': '-0+00-+00+-0+0+00+00-++-+00+000+0+0-0++-00+0+0-+000+0-+00+00+00+',
	# 	'e': '0+0-0++-00+0+0-++00-+0+00+00+00+-0+00-+00+-0+0+0-0+00-+00+00+00+' }
	

	w = m['a']
	d = ''
	genLen = int((length - 1) / len(c['a'])) + 1
	
	i = 1
	while len(w) < genLen:
		w += m[w[i]]
		i += 1

	for i in range(genLen):
		d += c[w[i]]
		
	return d[:length]

def genWord(length, m, c={}):
	"""
	Generate the word using dictionary morphism m, and optionnal dictionary coding c.
	"""

	i = 1
	w = m['a']

	while len(w) < length:
		w = w + m[w[i]]
		i += 1

	# If we don't use coding function (like for Thue-Morse word), we return it now
	if c == {}:
		return w
	
	# Otherwise, we build the good word and return it ...
	else:
		s = ''
		for i in range(length):
			s += c[w[i]]
		
		return s


def IFker(seq, args):
	"""
	Function that compute Kernel iterations for word length.
	"""

	k = int(args['-k'][:len(args['-k']) - 1])
	length = int(args['-s'])  # Size of subsequences in the Kernel
	length_MAX = length * (k ** args['-z'])  # Max size of subsequences in the Kernel after iteration
	matKer = []  # Matrix that contain all Kernel for length word iteration (args['-z'])

	print('\n* * * Computing Kernel iterations * * * \n')

	# Compute Kernel for increasing size of subsequences in kernel (args['-z'] + 1 iteration)
	while length <= length_MAX:
		K = {}  # Dico that contain all subsequences in Kernel for this length iteration
		matrix = []  # Matrix that contain all Kernel cardinality for this length iteration
		i = 0
		
		# Compute Kernel for the i-th iteration (k**i)
		while i <= args['-i']:
			K = kernelIter(K, seq, i, k, length)  # Compute next iteration Kernel and add it to previous one.
			matrix.append(len(K))

			if '-IVerb' in args or args['-i'] == i:
				ShowListKer(K, k, i, args, length)  # Show the list of subsequences in the Kernel for this length iteration
		
			i += 1

		length *= k  # Increase word length for length iteration
		matKer.append(matrix)  # Save Kernel cardinality for this length iteration
		print('')

	
	if '-m' in args:
		print('\n* * * Kernel iteration matrix * * * \n')
		Matrix(matKer)
		print('  Kernel iteration for :')
		print('    (horizontal) Kernel power up to i =', args['-i'])
		print('    (vertical) Subsequence length from ', args['-s'],' to ',args['-s'] ,'*', k**args['-z'],'.\n', sep='')

def kernelIter(C, u, i, k, SeqSize):
	"""
	Compute the k-kernel of the sequence u up to i-th iteration, and store subsequences of length SeqSize in the Kernel:
		C: dict - take the previously computed Kernel and append it with new iteration
		i: int - Iteration to compute (k**i)
		k: int - k-Kernel
		u: str/list - sequence we compute the Kernel
		SeqSize: int - Size of subsequences in the Kernel (greater = more precise)
	"""

	# Each subsequences, one letter every k**i letters
	for j in range(k ** i):
		l = 0
		ch = ''  # The subsequences
		
		# Build each subsequences
		while (k ** i) * l + j < len(u) and len(ch) < SeqSize:
			ch += u[(k ** i) * l + j]
			l += 1

		if ch not in C:
			C[ch] = 1  # New subsequence
		else:
			C[ch] += 1  # Subsequence arleady exist

	return C

def ShowListKer(K, k, i, args, seqLen):
	"""
	Function that print the Kernel of d
		K : list - The Kernel,
		k : int - k-Kernel, for print,
		i : int - power of k computed in k-Kernel, for print,
		arg : dict.
	"""

	cardMax = 0  # Max size of Kernel
	length = args['-u']  # Size of subsequences to show

	# Count and show content of Kernel 
	for u in K:
		# If we have to show the content of the Kernel (-1 always show all content)...
		if '-IShow' in args and (args['-IShow'] == -1 or args['-IShow'] == i ) :
			if length != -1:
				print('  | ', str(K[u]), '*', u[:length], ' ...' * max(0, min(1, len(u) - length)))
			else:
				print('  | ', str(K[u]), '*', u)
		cardMax += K[u]
	
	# Print summary about Kernel 
	print('  `> ',k,'-Kernel (Iter = ', i, ' ; SeqLen = ', seqLen, ' ; Prefix = ', seqLen * (k ** i), ') : ', len(K), ' distinct subsequences on ', cardMax, ' (r=', round(len(K) / cardMax, 5), '). \n', sep='')

def Matrix(mat, Xscale=[]):
	"""
	Simple matrix printing function for better visualisation.
	Xscale is an optional list of X title.
	"""

	maxChar = int(len(str(mat[len(mat) - 1][len(mat[0]) - 1])))

	if Xscale != []:
		maxChar = max(maxChar, len(str(Xscale[len(Xscale) - 1])))

		print('  ', end='')
		for i in range(len(Xscale)):  
			case = str(Xscale[i]) + ' ' * maxChar
			print(case[:maxChar], end=' ')

		print('\n')

	i = 0
	for i in range(len(mat)):  # Number of length iteration z
		print('  ', end='')
		
		for j in range(len(mat[0])):  # Number of Kernel iteration i
			case = str(mat[i][j]) + ' ' * maxChar
			print(case[:maxChar], end=' ')

		print('')
	print('')


def IFpattern(seq, args):
	"""
	Search and print the morphism that generated the sequence, if founded.
	"""
	# Custom values for morphism finding
	maxIter = args['-FStep']  # Number of iterations to check if the morphism match
	wSizeMax = args['-FWord']  # Maximal length of word that take the hypotetic morphism
	mSizeMax = args['-FMap']  # Maximal number of time the image is greater than pre-image 
	result = False

	print('\n* * * Finding matching morphism * * * \n')
	# If word may be too small for arguments, raise a Warning
	if len(seq) < wSizeMax * mSizeMax:
		print('  Warning : Word length is',len(seq),'but length greater than',wSizeMax * mSizeMax,'is advised!\n')

	result, result2 = morphismDetectV2(seq, maxIter, wSizeMax, mSizeMax, '-FVerb' in args, '-FCode' in args)
	# result store the morphism, coded or not, and result2 is either the occurence of each antecedent, or the coding.

	# If result is an empty dico, then they are no result ...
	if result == {}:
		print('  Can\'t find a morphism that generate the selected word!')
	
	# ... otherwise, print the morphism result an the antecedent_list/coding
	else:
		print('  Morphism found:', result, '\n')

		print('  Antecedents (', len(result2), '):', sep='')
		for i in result2.keys():
			print(' ', result2[i], ':', i)
	
	print('')

def morphismDetect(seq, lenMax, wordSizeMax, mapSizeMax, verb=False):
	"""
	Try to find the morphism that generate the given sequence. 
	"""

	wordSize = 0
	goodMorphism = False

	while wordSize < wordSizeMax and not goodMorphism:
		wordSize += 1
		mapSize = wordSize

		while mapSize < wordSize * mapSizeMax and not goodMorphism:
			mapSize += wordSize

			continueToSearch = True
			morphism = {}
			preimages = {}
			images = {}
			i = 0

			if verb:
				print('\n  (Word Size = ', wordSize, ', Map Size = ', mapSize, ')', sep='')

			while continueToSearch and i < lenMax:

				c = str(seq[i * wordSize:(i + 1) * wordSize]) # Readed pre-image of morphism m
				d = str(seq[i * mapSize:(i + 1) * mapSize]) # Hypotetic image of c by morphism m

				if c not in morphism:
					morphism[c] = d
					preimages[c] = 1

					if d not in images:
						images[d] = 1
					else:
						images[d] += 1
				
				elif c in morphism and morphism[c] != d:
					continueToSearch = False
					preimages[c] += 1

				else:
					preimages[c] += 1

					if d not in images:
						images[d] = 1
					else:
						images[d] += 1

				if verb:
					print('  [step = ',i+1,'/',lenMax,' | occ. = ',preimages[c],'] ',continueToSearch*'Continue' + (1-continueToSearch)*'SKIP    ',' >=> ',c, ' -> ',d, sep='')

				i += 1

			if i == lenMax and len(images) == len(preimages):
				goodMorphism = True
			
			elif i == lenMax :
				print('  Can match but Card(antecedent) =',len(preimages),'is not equal to Card(image) =',len(images),'!')
	
	if goodMorphism:
		return morphism, preimages
	else:
		return False, False, False

def morphismDetectV2(seq: str, maxIter: int, wordSizeMax: int, mapSizeMax: int, verb=False, makeCoding=False):
	"""
	Try to find the morphism that generate the given sequence. 'maxIter' is the number of time the function ckech if morphism is good, 'wordSizeMax' is the maximal word size the morphism can take, and 'mapSizeMax' specify the maximal morphism uniformity. 'verb' print all informations the function use.
	This version V2 find errors fastly than the previous one.
	"""

	wordSize = 0
	goodMorphism = False

	# Greater word size (antecedent) by +1 steps
	while wordSize != wordSizeMax and not goodMorphism:
		wordSize += 1
		mapSize = wordSize

		# Greater image size by + wordSize steps
		while mapSize != wordSize * mapSizeMax and not goodMorphism:
			mapSize += wordSize

			continueToSearch = True
			morphism = {}  # Morphism returned
			occurAnt = {}  # Occurence of antecedent
			i = 0

			if verb:
				print('\n  (Word Size = ', wordSize, ', Map Size = ', mapSize, ')', sep='')

			# While no error and word enough tall to see his image 'd' in it ...
			while continueToSearch and i * mapSize < len(seq):

				c = seq[i * wordSize:(i + 1) * wordSize] # Antecedent of morphism m
				d = seq[i * mapSize:(i + 1) * mapSize] # Hypotetic image of c by morphism m

				# If we've never seen c before, we check if the morphism is bijective
				if c not in morphism:
					continueToSearch = bool(d not in morphism.values())

					# If the morphism is bijective...
					if continueToSearch:
						morphism[c] = d  # Update the morphism
						occurAnt[c] = checkMorphismEntry(seq, c, d, i * wordSize, maxIter)  # Number of occurences of c

						# If c have more than 1 distinct image, occurAnt[c] < 0, and skip this morphism
						continueToSearch = bool(max(0,occurAnt[c]))  

					if verb:
						print('  ',continueToSearch*'OK   ' + (1-continueToSearch)*'SKIP ',' >=> ',occurAnt[c],' * ',c, ' --> ',d, sep='')

				i += 1
			
			# If the morphism is good and bijective, then stop searching 
			if continueToSearch:
				goodMorphism = True

	if goodMorphism:
	    
	    # If wordSize is big and we can code it with 62 letters, we code the morphism for better readability...
	    if wordSize > 2 and len(morphism) < 62 and makeCoding:
	        return codedMorphism(morphism, wordSize, mapSize)
	    
	    # Else we just return the result
	    else:
		    return morphism, occurAnt
	
	else:
		return {}, {}

def checkMorphismEntry(seq, c, d, start, maxIter=-1):
	"""
	Check if other occurence of word 'c' in sequence 'seq' have the same image 'd'.
	'maxIter' is the number of check it and 'start' specify the position of the word c in seq.
	If all occurence of c have the same image, return the number of occurences, otherwise return the negavite value (allow to know which image is not good).
	"""

	match = 1  # If match == 0, then c antecedent have two distinct images => return 0
	wordSize = len(c)
	mapSize = len(d)
	unif = int(mapSize / wordSize)  # Number of times images is greater than antecedent
	
	j = 0
	# While they are only one imag, still left test to do, c appears further ...
	while match > 0 and j != maxIter and start != -1:

		# search the new occurence of c in seq
		# Why start + wordSize - (start % wordSize) ? Because it search the 'good word' after the one readed : an antecedent is every wordSize characters, from the beginning
		# If the occurence of c found is not a 'good word', (start % wordSize) will be different from 0
		start = seq.find(c, start + wordSize - (start % wordSize))  

		# If the new occurence of c is found, if it is a 'good word' (not overlaping two word) and it's image is in seq length
		if start != -1 and not start % wordSize and start * unif + mapSize <= len(seq):

			# If the image is d, then one more occurence was found, otherwise one antecedent have two images
			if d == seq[start * unif:start * unif + mapSize]:	
				match += 1
			else:
				match *= -1

			j += 1
			

	return match

def codedMorphism(morphism, wordSize, mapSize):
	"""
	Code the morphism to convert antecedent from {+,0,-}-string to (a-z, 0-9, A-Z)-letters, return it, and return also the coding used.
	Usefull if antecedent are of form '+0+0-+0+000-++0-+-0+00+00-0++-0+000+00+0-+000+000+00000-0++0-+0-'.
	"""

	charMap = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	codedMorph = {}  # Morphism coded
	coding = {}  # Coding used
	i = 0
	
	# For each antecedent, we give it a letter in a-z, 0-9, A-Z
	for a in morphism.keys():
		coding[charMap[i]] = a
		i += 1

	# And then we create the coded morphism ...
	i = -1
	for b in morphism.values():  # For each map in the morphism ...
		i += 1
		for j in range(mapSize):  # For each word generated by an antecedent ...
			for e, f in coding.items():  # For each coded antecedent ...
				if b[j * wordSize:(j+1) * wordSize] == f:  # If the word generated match the coded antecedent, we update the coded morphism
					if j == 0: codedMorph[charMap[i]] = e
					else: codedMorph[charMap[i]] += e
    
	return codedMorph, coding


def IFcomplexity(seq, args):
	"""
	Give complexity of the sequence 'seq', up to specified value '-c' in args.
	"""
	
	print('\n* * * Computing complexity * * * \n')
	
	c = int(args['-c'][:len(args['-c']) - 1])
	step = int(len(seq) / args['-CStep'])

	Complexity = complexityCompute(seq, c, step)

	Matrix(Complexity)
	print('  Complexity iteration for :')
	print('    (horizontal) Sequence length from i = ', step,' to ',step,'*',args['-CStep'],'.',sep='')
	print('    (vertical) Bloc size from 1 to ', c,' .\n\n',sep='')

def complexityCompute(seq, blocSizeMax, step):
	"""
	Function that compute the complexity in sequence 'seq', up to size word 'BlocSizeMax'. 'step' say to compute complexity for diferent prefix length (every 'step' length).
	"""

	Complexity = []  # List of list of i-th complexity

	# Iteration for all size word (blocs) between 1 and max size specified
	for i in range(1, blocSizeMax + 1):
		j = 0
		cmplx = []

		length = step  # Iterate on diferent sequence length to see evolution
		subWord = []  # For a same bloc size, store all of them of size i
		# subWord is reset only when we change bloc size, and not durigng length iteration
		
		# Iteration for diferent sequence length
		while length <= len(seq):

			while j + i <= length:
				c = seq[j:j + i]  # Bloc considered

				if c not in subWord:
					subWord.append(c)

				j += 1

			length += step
			cmplx.append(len(subWord))
		
		Complexity.append(cmplx)
	
	return Complexity




if __name__ == "__main__":
	"""
	Main function that compute the PPL and calculate the kernel.
	Type '>>> python AStools.py man' to see manual.
	"""
	print('#'*75,'\n','#'*20,' STARTING Automatic Sequence Tools ','#'*20,'\n','#'*75,'\n',sep='')

	# Entry data :
	m, c, args = arguments(sys.argv[1:])  # For example, >>> python AStool.py ab,ba -n64 -k2w
	
	# Show the list of arguments if '--args' is in the dico(args) :
	if '--args' in args: print('Arguments:',m ,c , args,'\n')
	
	# Genere the infinite word w :
	w = genWord(int(args['-n']), m, c)
	if '-w' in args: print('  w   =', w[:args['-w']])

	# Compute the palindromic length of w :
	if '-p' in args or '-d' in args:
		PPL, d = palindromicLengths(w)  # Compute the PPL sequence and d(n) = PPL(n+1)-PPL(n) using PPL computing palindromicLengths
		d = dToStr(d)

		# If we wan't to print some sequences computed ...
		if '-p' in args: print('  PPL =', PPL[:args['-p']])
		if '-d' in args: print('  d   =', d[:args['-d']])
	else:
		PPL = d = []

	print('')

	# Compute the Kernel :
	if '-k' in args:
		seq = {'w': w, 'p': PPL, 'd': d}[args['-k'][len(args['-k']) - 1]]  # Select the sequence we compute the Kernel
		IFker(seq, args)

	# Try to find morphism pattern in d(n) sequence :
	if '-f' in args:
		IFpattern(dToStr(d), args)
	
	# Compute complexity of selected sequence :
	if '-c' in args:
		seq = {'w': w, 'p': PPL, 'd': d}[args['-c'][len(args['-c']) - 1]]  # Select the sequence we compute the Kernel
		IFcomplexity(seq, args)



# SPECIAL SECTION for test :
	# m = 1
	# n = 1
	# i = 0
	# while i < len(PPL)/27:
	# 	print(n - 1, ':', PPL[n - 1], '  ', m - 1, ':', PPL[m - 1], '  =  ', PPL[n - 1] == PPL[m - 1])
	# 	i += 1
	# 	n = 9 * i + 1
	# 	m = i

	# e = gen_d(args['-n'])
	# print(len(e), len(d))
	# print(e[:100],d[:100])
	# print(e==d)
# END Special section

