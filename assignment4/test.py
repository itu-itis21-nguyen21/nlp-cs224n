import torch

def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded = []

    ### YOUR CODE HERE (~6 Lines)

    # Find the length of the longest sentence
    max_len = len(max(sents, key=len))

    # Pad shorter sentences
    for sent in sents:
        number_of_pads = max_len - len(sent)
        sent.extend([pad_token] * number_of_pads)
        sents_padded.append(sent)

    ### END YOUR CODE

    return sents_padded

x = torch.randn(3, 4)
x.unsqueeze_(1)
print(x.size())