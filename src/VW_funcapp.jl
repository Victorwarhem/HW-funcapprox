module VW_funcapp
 
 
 	using FastGaussQuadrature
 	using PyPlot
 	import ApproXD: getBasis, BSpline
 	using Distributions
 	using ApproxFun
 	using CompEcon
 
 	
 	

 
 	function q1(n)

 		# The function to approximate:
 		f(x) = x .+ 2x.^2 - exp(-x)

 		# The interval:
 		a,b = (-3.0,3.0)

 		# The nodes:
 		z = gausschebyshev(n)

 		T(x,n) = cos(acos(x)*(n-1))
 		nodes = Float64[(z[1][i]+1)*(b-a) / 2 + a for i=1:n]

 		# The true function value
		y = f(nodes)

		# The basis function:
		Phi = Float64[cos((n-i+0.5)*(j-1)*pi/n) for i=1:n,j=1:15]

		# The coefficients:
		c = Phi \ y

		# We redifine the basis function with n=50
		n_new = 50
		
		nodesbis = linspace(a,b,n_new) 
		ytrue = f(nodesbis)
	
		map(x,a,b) = 2.*(x.-a)/(b.-a) - 1

		Phibis = Float64[T(map(nodesbis[i],a,b),j) for i=1:n_new,j=1:15]
		approx = Phibis * c

		# Here the error we find:
    	residual = approx - ytrue
    
    	# plot
    	plot(1:n_new;residual)
    	title("Deviation from true value")

   		println("The error is on average $(sum(residual)/n)") 
	end

	function q2(n)

		x = Fun(Interval(-3,3.0))
		g = x + 2x^2 - exp(-x)
		ApproxFun.plot(g; title="Question 2")
	end

	# plot the first 9 basis Chebyshev Polynomial

	function q3(n)

		# the number of graphs
		l=9

		# The function to grah
		Phi = Float64[cos((n-i+0.5)*(j-1)*pi/n) for i=1:n,j=1:l]

		# The plots
		fig,axes = subplots(3,3,figsize=(10,5))

		for i in 1:3
		for j in 1:3
		ax = axes[j,i]

		# To have the graphs rightly numbered
		nb = i+(j-1)*3

		ax[:plot](Phi[:,count])
		ax[:set_title]("Basis $(nb-1)")
		ax[:xaxis][:set_visible](false)
		ax[:set_ylim](-1.1,1.1)
		ax[:xaxis][:set_major_locator]=matplotlib[:ticker][:MultipleLocator](1)
 		ax[:yaxis][:set_major_locator]=matplotlib[:ticker][:MultipleLocator](1)
			end
		end
		return fig
	end

	# We associate a type and a function to ease our life for the next approximations.

	type ChebyType

		f::Function  
		nodes::Union{Vector,LinSpace} 
		basis::Matrix 
		coefs::Vector

		deg::Int 	
		lb::Float64
		ub::Float64

		function ChebyType(_nodes::Union{Vector,LinSpace},_deg,_a,_b,_f::Function)

 		n = length(_nodes)
 		y = _f(_nodes)
 		_basis = Float64[T(map(_nodes[i],_a,_b),j) for i=1:n,j=0:_deg]
 		_coefs = _basis \ y

 		# create a ChebyComparer with those values
 		new(_f,_nodes,_basis,_coefs,_deg,_a,_b)
 		end
 	end
 	
 	# function to predict points using info stored in ChebyType
 	
 	function predict(Ch::ChebyType,x_new)
 
 		true_new = Ch.f(x_new)
 		basis_new = Float64[T(map(x_new[i],Ch.a,Ch.b),j) or i=1:length(x_new);j=0:Ch.deg]
 		basis_nodes = Float64[T(map(Ch.nodes[i],Ch.lb,Ch.ub),j) for i=1:length(Ch.nodes),j=0:Ch.deg]
 		preds = basis_new * Ch.coefs
 		preds_nodes = basis_nodes * Ch.coefs
 
 		return Dict("x"=> x_new,"truth"=>true_new, "preds"=>preds, "preds_nodes" => preds_nodes)
 	end
 

	function q4a(deg=(5,9,15),a=-5.0,b=5.0)
 
 		# Define the function to approximate
 		f(x) = 1./(1+25.*x.^2)
 		l = linspace(-5,5,1000) # Points to extrapolate
 
 		fig,axes = subplots(1,2,figsize=(10,5))
 
 		ax = axes[1,1]
 		ax[:set_title]("Chebyshev nodes")
 
 		for j in 1:3
 
 		## Change the noods definition depending on why approach you take
 		nodes = gausschebyshev(deg[j]+1)[1]*5
 
 		x = predict(ChebyType(nodes,deg[j]+1,a,b,f),l)["x"]
 		y1 = predict(ChebyType(nodes,deg[j]+1,a,b,f),l)["truth"]
 		y2 = predict(ChebyType(nodes,deg[j]+1,a,b,f),l)["preds"]
 
 		ax[:plot](x,y1) # True function
 		ax[:plot](x,y2)
 		ax[:set_ylim](-1.1,1.1)
    	end
  
 		ax = axes[2,1]
 		ax[:set_title]("Uniform nodes")
 
 		for j in 1:3
  		## Change the noods definition depending on why approach you take
 		nodes = linspace(-5,5,deg[j]+1)
 
 		x = predict(ChebyType(nodes,deg[j]+1,a,b,f),l)["x"]
 		y1 = predict(ChebyType(nodes,deg[j]+1,a,b,f),l)["truth"]
 		y2 = predict(ChebyType(nodes,deg[j]+1,a,b,f),l)["preds"]
 
 		ax[:plot](x,y1)
 		ax[:plot](x,y2)
 		ax[:set_ylim](-1.1,1.1)
 		end
 
 		fig[:canvas][:draw]()
 		
 	end
 
 
 	function q4b()
 		
 		f(x) = 1./(1+25.*x.^2)
 		l = linspace(-5,5,10000)
 		f_true = f(l)
 
 		# Equally spaced knots.
 		bs = BSpline(13,3,-5,5) # order 3 because we want a cubic Spline. 13 knots between -5 and 5.
 
 		# Get the coefficients
 		B = full(getBasis(collect(linspace(-5,5,65)),bs)) # get the basis functions
 		y = f(linspace(-5,5,65))
 		c = B \ y
 
 		# Simulate the  function
 		B = full(getBasis(collect(l),bs)) # get the basis functions
 		f_app1 = B*c
 		dev1 = f_true - f_app1
 
 		# Knots concentrated toward 0.
 		myknots = vcat(collect(linspace(-5,-1,3)), collect(linspace(-0.5,0.5,7)), collect(linspace(1,5,3)))
 		bs = BSpline(myknots,3)
 		f_knots = f(myknots) - f(myknots)
 
 		# Get the coefficients
 		B = full(getBasis(collect(linspace(-5,5,65)),bs)) # get the basis functions
 		y = f(linspace(-5,5,65))
 		c = B \ y
 
 		# Simulate the  function
 		B = full(getBasis(collect(l),bs)) # get the basis functions
 		f_app2 = B*c
 		dev2 = f_true - f_app2
 
 		# Plot
 		fig,axes = subplots(1,2,figsize=(10,5))
 
 		ax = axes[1,1]
 		ax[:set_title]("True function")
 		ax[:plot](l,f_true)
 
 		ax = axes[2,1]
 		ax[:set_title]("Deviation from true function")
 		ax[:plot](l,dev1)
 		ax[:scatter](myknots,f_knots)
 		ax[:plot](l,dev2)
 		fig[:canvas][:draw]()
  	end
  
 	function q5()
  
 		f(x) = sqrt(abs(x))
 		xnew = linspace(-1,1,1000)
 		f_true = f(xnew)
 
 		a,b = (-1.0,1.0)
 		deg = 3
 		eval_points = collect(linspace(-1,1,65))
 
 		## Uniform knot vector.
 		par = SplineParams(linspace(a,b,13),0,deg)

		# Get the coefficients
 		B = CompEcon.evalbase(par,eval_points)[1]
 		y = f(eval_points)
 		c = B \ y
 
 		# Simulate the  function
 		B = CompEcon.evalbase(par,collect(l))[1]
 		f_app1 = B*c
 		dev1 = f_true - f_app1
 		print("The first deviation : $(dev1)")
 
 		## Knot multiplicity at 0
 		par = SplineParams(vcat(collect(linspace(lb,0,6)),0,collect(linspace(0,ub,6)),0,deg)
 
 		# Get the coefficients
 		B = CompEcon.evalbase(par,eval_points)[1]
 		y = f(eval_points)
 		c = B \ y
 
 		# Simulate the  function
 		B = CompEcon.evalbase(par,collect(l))[1]
 		f_app2 = B*c
 		dev2 = f_true - f_app2
		 print("The second deviation : $(dev2)")
 
 		# Plot
 		fig,axes = subplots(1,3,figsize=(10,5))
 
 		ax = axes[1,1]
 		ax[:set_title]("True function")
 		ax[:plot](l,f_true)
 		ax[:set_ylim](0,1.2)
 
 		ax = axes[2,1]
 		ax[:set_title]("Approximations")
 		ax[:plot](l,f_app1)
 		ax[:plot](l,f_app2)
 
 		ax = axes[3,1]
 		ax[:set_title]("Error approximations")
 		ax[:plot](l,dev1)
 		ax[:plot](l,dev2)
 		fig[:canvas][:draw]()
 
 		println("Question 5")
 		println("You need 3 knots equal to 0 to get a discontinuity in this point")
 	end
 
   	# function to run all questions
 	function runall()
 		println("running all questions of HW-funcapprox:")
 		q1(15)
 		q2(15)
 		q3()
 		q4a()
 		q4b()
 		q5()
 	end
 end
 
