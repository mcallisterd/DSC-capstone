### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 5ce0f0f2-f5db-11ea-0c10-1da125abcb63
begin
	using Flux, Flux.Data.MNIST, Statistics
	using Flux: onehotbatch, onecold, logitcrossentropy
	using Printf, BSON
	using CUDAapi
	using Images
	using Random
end

# ╔═╡ 99e57fc0-01f2-11eb-1511-d184bd2e39e2
using Plots

# ╔═╡ aa020b12-ff3d-11ea-0a5e-e3faaf53d4cb
md"""
# Handwritten LaTeX model

### First, basic imports and functions

These are the machine learning tools from Flux and then some tools to ensure we can use the GPU, display our data, and save our model.

Next, some helper functions that are doing basic things, usually in one line. tuplestopairs does what it says, takes an array of tuples and returns separately an array of the first elements of each tuple, and an array of the second elements.

augment() adds some noise to the data, we call it when we calculate the loss while training. I did not write this. It is from the Flux ML's "model zoo" example for solving MNIST with a simple CNN. I use the skeleton of their code there several times.

accuracy() does what you think, we have data, we have labels, we have a model. Evaluate the model on the data, compare it to the labels. Wrapping to making things fast and gpu friendly, again from Flux Model zoo example.

They have a safeguard against NaN parameters sneaking into the model, which needs two other helper functions. Hasn't ever been called when I have run their model. So haven't included yet.

And Arg() is an object type we just define here for helpfulness. It contains most of the hyperparameters, I think we include it because we need them all as parameters to various function calls throughout the process, and its faster to have them passed around, than have a global variable.

Oh and uniformsize() takes an image array and pads the top and bottom, left and right equally to ensure that all images are as large as the biggest one in the array. Pads with black.
"""

# ╔═╡ 278cf1a0-f5dc-11ea-306e-214592d200dc
if has_cuda()
    @info "CUDA is on"
    import CuArrays
    CuArrays.allowscalar(false)
end

# ╔═╡ 38288a80-feb8-11ea-24c7-81139212bd73
tuplestopairs(tuples) = map(x-> x[1],tuples),map(x->x[2],tuples)

# ╔═╡ 785c0320-ff3f-11ea-26fd-6381c2a36340
accuracy(x, y, model) = mean(onecold(cpu(model(x))) .== onecold(cpu(y)))

# ╔═╡ 7d355b22-ff40-11ea-3b3c-bb90182f17fa
md"""

### Larger "helper" functions


These are a function that gets all of the data, and repackages it nicely, getdata(), which uses the minibatch() function, which makes minibatches, predictably. Last is the build() function, it specifies how many layers we have of what sizes. 

Inside build, for reference, we have convolutional layer -> max pool -> convolutional -> max pool -> ... -> max pool -> fully connected, flat, relu -> output. The number of layers is subject to experimentation. Will see.
"""

# ╔═╡ e891fed0-feb1-11ea-07b6-a9132e344041
begin
	names =["Delta","implies","leftarrow","mathC","mathR","mathZ","nu","oplus","otimes","pm","rightarrow","times","cdot","equiv","exists","forall","alpha","beta","big_pi","big_sigma","cap","cdots","cup","epsilon","equals","gamma","geq","great","infty","lambda","leq","less","mini_delta","minus","neq","partial","phi","pi","plus","psi","rho","sigma","simeq","subset","supset","tau","theta","varepsilon","varphi","vee","wedge"]
	
	function loadAll()
		L = []
		for i ∈ 1:length(names)
			L = [L; loadAll(names[i],i)]
		end
		L
	end

	function loadAll(name,num)
		L,A,i= [],[],1
		while true
			try
				push!(L,(load("./subpictures/$name/$i.png"),num))
			catch e
				break
			end
			i+=1
		end
		L
	end
end

# ╔═╡ 6d76e9d0-01ff-11eb-03e0-69ebc7a2b1ff
@bind t html"""
<canvas width="87" height="87" style="position: relative"></canvas>
<script>
// 🐸 `this` is the cell output wrapper - we use it to select elements 🐸 //
const canvas = this.querySelector("canvas")
const ctx = canvas.getContext("2d")
var startX = 0
var startY = 0

ctx.fillStyle = '#000000'
ctx.fillRect(0, 0, 200, 200)
canvas.value= ""
function onmove(e){
	// 🐸 We send the value back to Julia 🐸 //
	canvas.value += [-e.layerX*100,e.layerY*100]
	canvas.dispatchEvent(new CustomEvent("input"))
	
	ctx.fillStyle = '#ffffff'
    ctx.fillRect(startX,startY, 2,2)
	startX = e.layerX
	startY = e.layerY
}
canvas.onmousedown = e => {
	ctx.beginPath();
	startX = e.layerX
	startY = e.layerY
	canvas.onmousemove = onmove
}
canvas.onmouseup = e => {
	canvas.onmousemove = null
	ctx.closePath()
}

</script>
"""

# ╔═╡ af33119e-01ff-11eb-39c5-430cbf08d9d2
t

# ╔═╡ d368c87e-01ff-11eb-1ff4-1f742e09bb54
Y = parse.(Int,filter(x-> length(x)>0,split(t,[',','-'])))[2:end-1]

# ╔═╡ 316d6300-0200-11eb-3bd1-ed6f63101398
function fourD2P(arr)
	pic = zeros(Int64,29,29)
	for k in 1:2:length(arr)
		x,y = ceil(Int64,arr[k]/300),ceil(Int64,arr[k+1]/300)
		pic[x,y] = 1
	end
	Gray.(pic)
end

# ╔═╡ 6ea22df0-0200-11eb-347b-330c965c8d62
J = fourD2P(Y)

# ╔═╡ bea0bc80-0201-11eb-2dce-d7d09da47851
Float32.(J)

# ╔═╡ 9ee3d3b0-0129-11eb-3eac-c9ffffc6d762
function minibatch(data,labels,vals)
	data_batch = Array{Float32}(undef,size(data[1])...,1,length(vals))#Initialize
	for (i,k) ∈ enumerate(vals)
		data_batch[:,:,:,i] = Float32.(data[k])
	end
	label_batch = onehotbatch(labels[vals],1:length(names))
	return data_batch,label_batch
end

# ╔═╡ bc8a76d0-feb8-11ea-1866-8f3f047d2960
function getdata()
	img,lab = tuplestopairs(shuffle(loadAll()))
	cutoff = ceil(Int,4length(img)/5)
	train_images,train_labels = img[1:cutoff],lab[1:cutoff]
	test_images,test_labels = img[cutoff:end],lab[cutoff:end]
	train_set = [minibatch(train_images, train_labels, k:min(k+128,length(train_images))) for k ∈1:128:length(train_images)]
   	test_set = minibatch(test_images, test_labels, 1:length(test_images))
   	return train_set, test_set
end

# ╔═╡ 461e1610-ff43-11ea-0777-559c10ef2898
function build()
	return Chain(
		Conv((3,3), 1 => 16, pad = (1,1), relu),
		MaxPool((2,2)),
		Conv((3,3), 16 => 32, pad = (1,1), relu),
		MaxPool((2,2)),
		Conv((3,3), 32 => 32, pad = (1,1), relu),
		MaxPool((2,2)),
		flatten,
		Dense(prod(floor.(Int,[29/8,29/8, 32])),51) 
		)
end

# ╔═╡ c9230580-0127-11eb-2fa1-ef2c5bac8a60
function train(go)
	training,testing = getdata()
	model = build()
	training,testing,model = gpu.(training),gpu.(testing),gpu(model)
	model(training[1][1])
	optimiser = ADAM(3e-3)
		loss(x,y) = logitcrossentropy(model(x),y)
		top_score,best_mod = 0,missing
	for k ∈ 1:20
		Flux.train!(loss, params(model), training, optimiser)
		score = accuracy(testing..., model)
		@info(@sprintf("[%d]: Test accuracy: %.4f", k, score))
		if score >= top_score
			best_mod = deepcopy(model)
			top_score = score
		end
	end
	best_mod,top_score
end


# ╔═╡ fd9ff6ee-01f0-11eb-0f7c-e339d3a13a91
a,b = getdata();

# ╔═╡ 6ed7b022-01f0-11eb-1e37-8ffceddc9d99
function what(M)
	A,B = gpu.(a),gpu.(b)
	M(A[1][1])
end

# ╔═╡ dc25c120-01f1-11eb-2700-fdf3a3abfb23
function were(M)
	M(a[1][1])
end

# ╔═╡ f7b27a70-012c-11eb-2a5d-3d9e1f40a894
A,B = train(2)

# ╔═╡ e7475550-01f1-11eb-1ef6-8f4b9e615d1d
P = cpu(A)

# ╔═╡ ee66c230-01f1-11eb-10c9-df29ee1bdeb8
S = were(P)

# ╔═╡ a1faa930-01f0-11eb-29e9-9f24b958858b
predict(mat,k) = findall(x-> (x-maximum(mat[:,k])==0),mat[:,k])[1]

# ╔═╡ 6e76d600-01f1-11eb-11ad-c3531927f4da
[predict(S,k) for k∈1:10]

# ╔═╡ 74027190-01f3-11eb-11af-039b06e83dad
gr()

# ╔═╡ 44a4da60-01f2-11eb-1e96-2d6308c6c7d7
[names[predict(S,k)] for k∈1:20]

# ╔═╡ 9dd19190-01f3-11eb-2745-73993ec69ada
o = [Gray.(a[1][1][:,:,1,k]) for k ∈1:20]

# ╔═╡ d99053c0-01f2-11eb-0c37-6d1596d0cb8d
names[41]

# ╔═╡ 29973e60-0202-11eb-295b-a1ab9163e8e5
function wrapper(img)
	X = zeros(Float32,size(img)...,1,1)
	X[:,:,1,1] = img
	X
end

# ╔═╡ 4fb82370-0202-11eb-096c-db86514ee411
K = wrapper(Float32.(J))

# ╔═╡ 087d4670-0202-11eb-2f77-bb3e7b8d2a47
names[predict(P(K),1)]

# ╔═╡ Cell order:
# ╟─aa020b12-ff3d-11ea-0a5e-e3faaf53d4cb
# ╠═5ce0f0f2-f5db-11ea-0c10-1da125abcb63
# ╠═278cf1a0-f5dc-11ea-306e-214592d200dc
# ╠═38288a80-feb8-11ea-24c7-81139212bd73
# ╠═785c0320-ff3f-11ea-26fd-6381c2a36340
# ╟─7d355b22-ff40-11ea-3b3c-bb90182f17fa
# ╠═e891fed0-feb1-11ea-07b6-a9132e344041
# ╠═bc8a76d0-feb8-11ea-1866-8f3f047d2960
# ╠═6d76e9d0-01ff-11eb-03e0-69ebc7a2b1ff
# ╠═af33119e-01ff-11eb-39c5-430cbf08d9d2
# ╠═d368c87e-01ff-11eb-1ff4-1f742e09bb54
# ╠═6ea22df0-0200-11eb-347b-330c965c8d62
# ╠═bea0bc80-0201-11eb-2dce-d7d09da47851
# ╠═316d6300-0200-11eb-3bd1-ed6f63101398
# ╠═9ee3d3b0-0129-11eb-3eac-c9ffffc6d762
# ╠═461e1610-ff43-11ea-0777-559c10ef2898
# ╠═c9230580-0127-11eb-2fa1-ef2c5bac8a60
# ╠═fd9ff6ee-01f0-11eb-0f7c-e339d3a13a91
# ╠═6ed7b022-01f0-11eb-1e37-8ffceddc9d99
# ╠═dc25c120-01f1-11eb-2700-fdf3a3abfb23
# ╠═f7b27a70-012c-11eb-2a5d-3d9e1f40a894
# ╠═e7475550-01f1-11eb-1ef6-8f4b9e615d1d
# ╠═ee66c230-01f1-11eb-10c9-df29ee1bdeb8
# ╠═a1faa930-01f0-11eb-29e9-9f24b958858b
# ╠═6e76d600-01f1-11eb-11ad-c3531927f4da
# ╠═99e57fc0-01f2-11eb-1511-d184bd2e39e2
# ╠═74027190-01f3-11eb-11af-039b06e83dad
# ╠═44a4da60-01f2-11eb-1e96-2d6308c6c7d7
# ╠═9dd19190-01f3-11eb-2745-73993ec69ada
# ╠═d99053c0-01f2-11eb-0c37-6d1596d0cb8d
# ╠═087d4670-0202-11eb-2f77-bb3e7b8d2a47
# ╠═29973e60-0202-11eb-295b-a1ab9163e8e5
# ╠═4fb82370-0202-11eb-096c-db86514ee411
