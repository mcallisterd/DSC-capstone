### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ d21f0b40-01cf-11eb-17c8-bf5508994f0f
begin
	using Images
	using ImageIO
	using PlutoUI
end

# ╔═╡ f909d000-01cf-11eb-071d-8b2ef5ddd63f
decimate(img,t) = img[1:t:end,1:t:end]

# ╔═╡ 004661d0-01d0-11eb-0105-d9cee9ad4c58
normalload(name) = load("./subpictures/$name.png")

# ╔═╡ 15bd3980-01d0-11eb-0ad4-6f49a9a3a07b
function killBR(p)
	if sum(convert.(Float32,[p.g,p.b])) > 1.5 || p.r>.7
		RGB(0,0,0)
	else
		RGB(1-p.r,1-p.g,1-p.b)
	end 
end

# ╔═╡ 7b38ba40-01d1-11eb-3d5f-9b99f3e9b1bb
S= ["Delta","implies","leftarrow","mathC","mathR","mathZ","nu","oplus","otimes","pm","rightarrow","times","cdot","equiv","exists","forall","alpha","beta","big_pi","big_sigma","cap","cdots","cup","epsilon","equals","gamma","geq","great","infty","lambda","leq","less","mini_delta","minus","neq","partial","phi","pi","plus","psi","rho","sigma","simeq","subset","supset","tau","theta","varepsilon","varphi","vee","wedge"]

# ╔═╡ 2e5b41d0-01d0-11eb-00ad-1996c835d86c
washedload(name) = load("./subpictures/washed_$name.png")

# ╔═╡ 5da371b0-01d0-11eb-248b-27eba61bc10c
level = .4

# ╔═╡ 5aabdc8e-01d0-11eb-3aa9-29f89d303ee9
polarize(x) =  x>level ? 1 : 0

# ╔═╡ 1aa86140-01d0-11eb-0be4-ddf45497bde5
function noLines(name)
	save("./subpictures/washed_$name.png",decimate(Gray.(polarize.(Gray.(killBR.(normalload(name))))),3))
end

# ╔═╡ 5f501ad0-01d1-11eb-334d-df9c10e3ec61
function washall()
	for name in S
		noLines(name)
	end
end

# ╔═╡ dfbfdad0-01d0-11eb-0958-c729cd56fc34
isblank(img)  = sum(Float32.(img))<10

# ╔═╡ a2681e8e-01d0-11eb-3c5d-13369202bfa3
function split(img,width,height,name)
	L,B = size(img)
	W,H,i = 1,1,1
	while W<B
		while H<L
			spot = "./subpictures/$name/$i.png"
			if H+height>L && W+width>B
				sub = img[H:end,W:end]
			elseif H+height>L
				sub = img[H:end,W:W+width]
			elseif W+width>B
				sub = img[H:H+height,W:end]
			else
				sub = img[H:H+height,W:W+width]
			end
			if prod(size(sub))>= 600 && (!isblank(sub) || name[1:2]=="cd" || name =="minus" || name == "times")
				save(spot,sub)
				i+=1
			end
			H+=height
		end
		H=1
		W+=width
	end
end

# ╔═╡ ed47727e-01d0-11eb-2f62-a91f78f0cee1
function washsplit(a,b)
	for name in S
		split(washedload(name),a,b,name)
	end
end

# ╔═╡ a5d3ffb2-01d8-11eb-0254-ed59663929e9
function pad(I)
	s = size(I)
	if s[1]<29
		x = (29-s[1])/2
		I = [zeros(Gray,ceil(Int,x),s[2]); I; zeros(Gray, floor(Int,x),s[2])]
	end
	if s[2]<29
		y = (29-s[2])/2
		I = [zeros(Gray,size(I)[1],ceil(Int,y)) I zeros(Gray, size(I)[1],floor(Int,y))]
	end
	I
end

# ╔═╡ 644c71e0-01d7-11eb-1772-b74f8e9b953f
begin
	function loadAll(name,num)
		L,i= [],1
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
	
	function loadAll()
		L = []
		for k in 1:length(S)
			L = [L; loadAll(S[k],k)]
		end
		L
	end
end

# ╔═╡ 3efa7160-01de-11eb-2060-d1535e19eee4
function update(name)
	i = 1
	while true
		try
			I = load("./subpictures/$name/$i.png")
			save("./subpictures/$name/$i.png",pad(I))
		catch e
			break
		end
		i+=1
	end
end

# ╔═╡ 8b916060-01de-11eb-2450-addd72ed6530
function updateAll()
	for name in S
		update(name)
	end
end

# ╔═╡ Cell order:
# ╠═d21f0b40-01cf-11eb-17c8-bf5508994f0f
# ╠═f909d000-01cf-11eb-071d-8b2ef5ddd63f
# ╠═004661d0-01d0-11eb-0105-d9cee9ad4c58
# ╠═15bd3980-01d0-11eb-0ad4-6f49a9a3a07b
# ╠═1aa86140-01d0-11eb-0be4-ddf45497bde5
# ╠═7b38ba40-01d1-11eb-3d5f-9b99f3e9b1bb
# ╠═5f501ad0-01d1-11eb-334d-df9c10e3ec61
# ╠═2e5b41d0-01d0-11eb-00ad-1996c835d86c
# ╠═5da371b0-01d0-11eb-248b-27eba61bc10c
# ╠═5aabdc8e-01d0-11eb-3aa9-29f89d303ee9
# ╠═dfbfdad0-01d0-11eb-0958-c729cd56fc34
# ╠═ed47727e-01d0-11eb-2f62-a91f78f0cee1
# ╠═a2681e8e-01d0-11eb-3c5d-13369202bfa3
# ╠═a5d3ffb2-01d8-11eb-0254-ed59663929e9
# ╠═644c71e0-01d7-11eb-1772-b74f8e9b953f
# ╠═8b916060-01de-11eb-2450-addd72ed6530
# ╠═3efa7160-01de-11eb-2060-d1535e19eee4
