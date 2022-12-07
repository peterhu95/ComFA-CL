from helper import *

class GRL(torch.autograd.Function):
    scale = 1.0
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return GRL.scale * grad_output.neg()
            
def grad_reverse(x, scale=1.0):
    GRL.scale = scale
    return GRL.apply(x)

class ConvE(torch.nn.Module):
	"""
	Proposed method in the paper. Refer Section 6 of the paper for mode details 

	Parameters
	----------
	params:        	Hyperparameters of the model
	
	Returns
	-------
	The AcrE model instance
		
	"""
	def __init__(self, params):
		super(ConvE, self).__init__()

		self.p                  = params
		self.ent_embed		= torch.nn.Embedding(self.p.num_ent,   self.p.embed_dim, padding_idx=None); xavier_normal_(self.ent_embed.weight)
		self.rel_embed		= torch.nn.Embedding(self.p.num_rel*2, self.p.embed_dim, padding_idx=None); xavier_normal_(self.rel_embed.weight)
		self.bceloss		= torch.nn.BCELoss()

		self.inp_drop		= torch.nn.Dropout(self.p.inp_drop)
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.feature_map_drop	= torch.nn.Dropout2d(self.p.feat_drop)
		self.bn0 = torch.nn.BatchNorm2d(1)
		self.bn1 = torch.nn.BatchNorm2d(32)
		self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)
		self.fc = torch.nn.Linear(9728, self.p.embed_dim)
		
		self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=self.p.bias)
			
		self.register_parameter('b', Parameter(torch.zeros(self.p.num_ent)))
   
		self.rm_fea = None

		self.classifier = torch.nn.Linear(self.p.embed_dim, self.p.num_ent)

	def loss(self, pred, true_label=None, sub_samp=None):
		label_pos	= true_label[0];
		label_neg	= true_label[1:]
		loss 		= self.bceloss(pred, true_label)
		return loss

	def forward(self, sub, rel, neg_ents, scale, strategy='one_to_x'):
		sub_emb		= self.ent_embed(sub).view(-1, 1, 20, 10)
		rel_emb		= self.rel_embed(rel).view(-1, 1, 20, 10)
		comb_emb	= torch.cat([sub_emb, rel_emb], dim=2)
		stack_inp = self.bn0(comb_emb)
		x		= self.inp_drop(stack_inp)
		x = self.conv1(x)
		x		= self.bn1(x)
		x		= F.relu(x)
		x		= self.feature_map_drop(x)
		x = x.view(x.shape[0], -1)
		x = self.fc(x)
   
		self.rm_fea = x

		x   = grad_reverse(x, scale)
   
		x		= self.hidden_drop(x)
		x		= self.bn2(x)
		x		= F.relu(x)

		if strategy == 'one_to_n':
			x = self.classifier(x)#torch.mm(x, self.ent_embed.weight.transpose(1,0))
			#x += self.b.expand_as(x)
		else:
			x = torch.mul(x.unsqueeze(1), self.ent_embed(neg_ents)).sum(dim=-1)
			x += self.b[neg_ents]

		pred	= torch.sigmoid(x)

		return pred


class InteractE2(torch.nn.Module):
	"""
	Proposed method in the paper. Refer Section 6 of the paper for mode details 

	Parameters
	----------
	params:        	Hyperparameters of the model
	chequer_perm:   Reshaping to be used by the model
	
	Returns
	-------
	The InteractE model instance
		
	"""
	def __init__(self, params, chequer_perm):
		super(InteractE2, self).__init__()

		self.p                  = params
		self.ent_embed		= torch.nn.Embedding(self.p.num_ent,   self.p.embed_dim, padding_idx=None); xavier_normal_(self.ent_embed.weight)
		self.rel_embed		= torch.nn.Embedding(self.p.num_rel*2, self.p.embed_dim, padding_idx=None); xavier_normal_(self.rel_embed.weight)
		self.bceloss		= torch.nn.BCELoss()

		self.inp_drop		= torch.nn.Dropout(self.p.inp_drop)
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.feature_map_drop	= torch.nn.Dropout2d(self.p.feat_drop)
		self.bn0		= torch.nn.BatchNorm2d(self.p.perm)

		flat_sz_h 		= self.p.k_h
		flat_sz_w 		= 2*self.p.k_w
		self.padding 		= 0

		self.bn1 		= torch.nn.BatchNorm2d(self.p.num_filt*self.p.perm)
		self.flat_sz 		= flat_sz_h * flat_sz_w * self.p.num_filt*self.p.perm

		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		self.fc 		= torch.nn.Linear(self.flat_sz, self.p.embed_dim)
		self.chequer_perm	= chequer_perm

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))
		self.register_parameter('conv_filt', Parameter(torch.zeros(self.p.num_filt, 1, self.p.ker_sz,  self.p.ker_sz))); xavier_normal_(self.conv_filt)
   
		self.conv_aggr = torch.nn.Conv2d(1, self.p.CFR_kernels, (2, 2), 1, 1, bias=True)
		self.cal_fc = torch.nn.Linear(self.p.CFR_kernels*3*201, self.p.embed_dim)
		self.cal_bn = torch.nn.BatchNorm2d(self.p.CFR_kernels)
   
		self.Ctrans = torch.nn.Linear(self.p.embed_dim, self.p.embed_dim)
		self.pos_trans = torch.nn.Linear(self.p.embed_dim, self.p.embed_dim)
		self.neg_trans = torch.nn.Linear(self.p.embed_dim, self.p.embed_dim)
		self.pos = None
		self.neg = None
		self.pos_drop=torch.nn.Dropout(0.1)
		self.neg_drop=torch.nn.Dropout(0.1)
		self.prelu = torch.nn.PReLU()
		self.discri = None
		self.C_info = None
		self.com_trans = torch.nn.Sequential(
                          torch.nn.Linear(self.p.embed_dim, self.p.embed_dim),
                          torch.nn.LeakyReLU()
                          )
		self.w_bigtopK = torch.nn.Linear(self.p.embed_dim, self.p.embed_dim)

        
	def sim(self, t1, t2, temperature):
		t1_norm = torch.norm(t1, dim=-1, keepdim=True)  # (128,200)
		t2_norm = torch.norm(t2, dim=-1, keepdim=True)  # (128,200)
		dot_numerator = torch.mm(t1, t2.t())  # (128,128)
		dot_denominator = torch.mm(t1_norm, t2_norm.t()) + 1e-8  # (128,128)
		sim_matrix = torch.exp(dot_numerator / dot_denominator / temperature)
		return sim_matrix
    
 
	def contra_loss(self, temperature=0.05):  # (128, 200)
		similarity = torch.nn.functional.normalize(self.discri).mm(torch.nn.functional.normalize(self.ent_embed.weight.data).transpose(0,1))
		#similarity_all = self.sim(self.discri, self.ent_embed.weight.data, temperature) # (128, 14543)
		similarity_all =torch.exp(similarity.div(temperature)) # (128, 14543)
		_, ind = similarity_all.topk(self.p.topK, dim=1)
		row_idx = torch.arange(similarity_all.size(0)).unsqueeze(1)
		mask=torch.ones_like(similarity_all)
		mask[row_idx,ind] = 0
		if not self.p.descending:
				add_mask = torch.where(mask==0, 1e+8, 0.0)
				_, ind_neg = (similarity_all.mul(mask) + add_mask).topk(int((self.ent_embed.num_embeddings-self.p.topK) *self.p.ratio), largest=False, dim=1)
		else:
				_, ind_neg = similarity_all.topk(int((self.ent_embed.num_embeddings-self.p.topK) *self.p.ratio), dim=1)


		similarity = torch.softmax(similarity.gather(1, ind), dim=-1)  # (128, topK)
		similarity = similarity.unsqueeze(2)  # (128, topK, 1)
		sim_emb = self.ent_embed.weight[ind.flatten(),:].view(self.discri.size(0), self.p.topK, -1)  # (128, topK, 200)
        
		#wei_emb = similarity * sim_emb
		#self.pos = wei_emb.sum(dim=1)  # (128, 200)  # postive
		self.pos = similarity.transpose(1,2).bmm(sim_emb).squeeze()  # (128, 200)  # postive
		self.pos = torch.nn.functional.leaky_relu(self.pos_trans(self.C_info + self.pos))
   
		matrix_dis2pos = self.sim(self.discri, self.pos, temperature).diag().view(-1, 1)  # (128, 1)
   
		matrix_dis2neg = similarity_all
		
		matrix_dis2neg = matrix_dis2neg.gather(1, ind_neg).sum(dim=1).view(-1, 1)  # (128, 1)
        
		matrix_contra = matrix_dis2pos/((matrix_dis2pos+matrix_dis2neg + 1e-8))
		logits = -torch.log(matrix_contra).mean()

		return logits
   

	def loss(self, pred, true_label=None, sub_samp=None):
		label_pos	= true_label[0]; 
		label_neg	= true_label[1:]
		loss 		= self.bceloss(pred, true_label)
		return loss

	def circular_padding_chw(self, batch, padding):
		upper_pad	= batch[..., -padding:, :]
		lower_pad	= batch[..., :padding, :]
		temp		= torch.cat([upper_pad, batch, lower_pad], dim=2)

		left_pad	= temp[..., -padding:]
		right_pad	= temp[..., :padding]
		padded		= torch.cat([left_pad, temp, right_pad], dim=3)
		return padded
   
	def conv_aggr_layer(self, fea, rm_fea):  #(bs, 200)
		rm_fea = self.com_trans(rm_fea)
		rm_fea = torch.nn.functional.normalize(rm_fea,p=2,dim=1)
		fea = torch.nn.functional.normalize(fea,p=2,dim=1)
		bs = fea.shape[0]
		x = torch.cat((fea, rm_fea), dim=1)  # (bs, 400)
		x = x.view(bs, 1, 2, self.p.embed_dim)
		x = self.conv_aggr(x)  # (bs, 16, 1, 200)
		x = self.cal_bn(x)
		x = F.relu(x)
		x = self.feature_map_drop(x)
		x = x.view(x.shape[0], -1)
		x = self.cal_fc(x)  # (bs, 200)
		return x

	def forward(self, sub, rel, neg_ents, rm_fea, strategy='one_to_x'):
		sub_emb		= self.ent_embed(sub)
		rel_emb		= self.rel_embed(rel)
		comb_emb	= torch.cat([sub_emb, rel_emb], dim=1)
		chequer_perm	= comb_emb[:, self.chequer_perm]
		stack_inp	= chequer_perm.reshape((-1, self.p.perm, 2*self.p.k_w, self.p.k_h))
		stack_inp	= self.bn0(stack_inp)
		x		= self.inp_drop(stack_inp)
		x		= self.circular_padding_chw(x, self.p.ker_sz//2)
		x		= F.conv2d(x, self.conv_filt.repeat(self.p.perm, 1, 1, 1), padding=self.padding, groups=self.p.perm)
		x		= self.bn1(x)
		x		= F.relu(x)
		x		= self.feature_map_drop(x)
		x		= x.view(-1, self.flat_sz)
		x		= self.fc(x)
   
		x		= self.hidden_drop(x)
		x		= self.bn2(x)
		x		= F.relu(x)
   
		self.discri = x
		self.C_info = torch.nn.functional.leaky_relu(self.Ctrans(rm_fea))  #(128, 200)

		if strategy == 'one_to_n':
			x = torch.mm(x, self.ent_embed.weight.transpose(1,0))
			x += self.bias.expand_as(x)
		else:
			x = torch.mul(x.unsqueeze(1), self.ent_embed(neg_ents)).sum(dim=-1)
			x += self.bias[neg_ents]

		pred	= torch.sigmoid(x)

		return pred