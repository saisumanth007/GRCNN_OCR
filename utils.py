def str2label(strs, max_len):

	def ascii_val2label(ascii_val):
		if ascii_val >= 48 and ascii_val <= 57:    # '0'-'9' are mapped to 1-10
			label = ascii_val - 47
		elif ascii_val >= 65 and ascii_val <= 90:  # 'A'-'Z' are mapped to 11-36
			label = ascii_val - 64 + 10
		elif ascii_val >= 97 and ascii_val <= 122: # 'a'-'z' are mapped to 11-36
			label = ascii_val - 96 + 10
		else:
			label = 0
		return label

	num_strings = len(strs)
	labels = [[0]*max_len for _ in xrange(num_strings)]
	for i, str1 in enumerate(strs):
		for j in xrange(len(str1)):
			ascii_val = ord(str1[j])
			labels[i][j] = ascii_val2label(ascii_val)
	return labels

def label2str(labels, raw):

	def label2ascii_val(label):
		if label >= 1 and label <= 10:
			ascii_val = label - 1 + 48
		elif label >= 11 and label <= 36:
			ascii_val = label - 11 + 97
		elif label == 0:
			ascii_val = ord('-')
		return ascii_val

	strs = []
	num_strings, max_len = len(labels), len(labels[0])
	for i in xrange(num_strings):
		str1 = ''
		labels_i = labels[i]
		for j in xrange(max_len):
			if raw:
				str1 += chr(label2ascii_val(labels_i[j]))
			else:
				if labels_i[j] == 0:
					break
				else:
					str1 += chr(label2ascii_val(labels_i[j]))
		strs.append(str1)
	return strs