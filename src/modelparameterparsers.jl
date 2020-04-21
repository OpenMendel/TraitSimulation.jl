#Find and return the variable names in the expression x. This function is called in the main function mean_formula.
find_variables(x) = find_variables!(Symbol[], x) #this is so that we can call the function without the exclamation
function find_variables!(var_names, x::Number) #if the variable name is a number then just return it without doing anything
    return var_names
end
function find_variables!(var_names, x::Symbol) # if the variable name is a symbol then push it to the list of var_names because its a name of a column
    push!(var_names, x)
end

function find_variables!(var_names, x::Expr) # if the variable is a expression object then we have to crawl through each argument of the expression
    # safety checking
    if x.head == :call  # check for + symbol bc we are summing linear combinations within each expression argument
      # pass the remaining expression
      for argument in x.args[2:end] # since the first argument is the :+ call
        find_variables!(var_names, argument) #recursively find the names given in each argument so check if number, if symbol, if expression etc.again agian
    end
end
return var_names
end

#Search the expression x for the variable name in var, and then create the julia interpretable expression. We extend this function to allow for recursion on the variable vars... This function is called in the main function mean_formula.
function search_variables!(x::Expr, var::Symbol)
    for i in eachindex(x.args)
        if x.args[i] == var # if the argument is one of the variables given then just put it in the right format df[:x1]
            x.args[i] = Meta.parse(string(:input_data_from_user,"[!, ", ":", var, "]"))
        elseif x.args[i] isa Expr # else if the argument is an expression (i.e not a varaible (symbol) or a number) then
            search_variables!(x.args[i], var) #go through this function recursively on each of the arguments of the expression object
        end
    end
    return x
end

function search_variables!(x::Expr, vars...) # this is for when you have more than one variable name found in the string
    for var in vars #goes through each of the variables in the vector vars
        x = search_variables!(x, var) #runs the recursion on each variable in vars
    end
    return x
end

"""
    mean_formula(user_formula_string::String, df::DataFrame)
Construction of the evaluated mean vector, given formula string and named dataframe of covariates as input. This function allows for transformations of any functions supported by julia.
"""
function mean_formula(user_formula_string::String, df::DataFrame)
    global input_data_from_user = df #this is so we can call whatever name the user has for the dataframe

    users_formula_expression = Meta.parse(user_formula_string)
    if(users_formula_expression isa Expr)
        found_markers = find_variables(users_formula_expression) #store the vector of symbols of the found variables
        #X = Matrix(df[:, found_markers)
        dotted_args = map(Base.Broadcast.__dot__, users_formula_expression.args) # adds dots to the arguments in the expression
        dotted_expression = Expr(:., dotted_args[1], Expr(:tuple, dotted_args[2:end]...)) #reformats the exprssion arguments by changing the variable names to tuples of the variable names to keep the dot structure of julia

        julia_interpretable_expression = search_variables!(dotted_expression, found_markers...) #gives me the julia interpretable exprsesion with the dataframe provided

        mean_vector = eval(Meta.parse(string(julia_interpretable_expression))) #evaluates the julia interpretable expression on the dataframe provided
    else
        mean_vector = [users_formula_expression for i in 1:size(df, 1)]
    end
    return mean_vector, found_markers
end

"""
    FixedEffectTerms(effectsizes::AbstractVecOrMat, snps::AbstractVecOrMat)
Construction of the proper String expression for evaluation in the simulation process,
 using the specified vectors of regression coefficients (Effect Sizes) as a vector of numbers and snp names as a vector of strings.
"""
function FixedEffectTerms(effectsizes::AbstractVecOrMat, snps::AbstractVecOrMat)
 # implementation
    fixed_terms = ""
    for i in 1:length(effectsizes)
        expression = " + " * string(effectsizes[i]) * "(" * snps[i] * ")"
        fixed_terms = fixed_terms * expression
    end

    return String(fixed_terms)
end

"""
append_terms
Allows us to append terms to create a VarianceComponent type
"""
function append_terms!(AB, summand)
	A_esc = esc(summand.args[2])	# elements in args are symbols,
	B_esc = esc(summand.args[3])
	push!(AB.args, :(VarianceComponent($A_esc, $B_esc)))
end

"""
this vc macro allows us to create a vector of VarianceComponent objects for simulation so with_bigfloat_precis, precision::Integer)
so that the user can type out @vc V[1] ⊗ Σ[1] + V[2] ⊗ Σ[2] + .... + V[m] ⊗ Σ[m]
"""
macro vc(expression)
	n = length(expression.args)
	AB = :(VarianceComponent[]) # AB is an empty vector of variance components list of symbols
	if expression.args[1] != :+ #if first argument is not plus (only one vc)
		summand = expression
		append_terms!(AB, summand)
	else #MULTIPLE VARIANCE COMPONENTS if the first argument is a plus (Sigma is a sum multiple variance components)
		for i in 2:n
			summand = expression.args[i]
			append_terms!(AB, summand)
		end
	end
	return(:($AB))
end

"""
vcobjectuple(vcobject)
This function creates a tuple of Variance Components, given a vector of variancecomponent objects to be compatible with VarianceComponentModels.jl
"""
function  vcobjtuple(vcobject::Vector{VarianceComponent})
	m = length(vcobject)
	d = size(vcobject[1].Σ, 1)
	n = size(vcobject[1].V, 1)
	Σ = ntuple(x -> zeros(d, d), m)
	V = ntuple(x -> zeros(n, n), m)
	for i in eachindex(vcobject)
		copyto!(V[i], vcobject[i].V)
		copyto!(Σ[i], vcobject[i].Σ)
	end
	return(Σ, V)
end

"""
This is a wrapper linear algebra function that computes the fixed effects. [C1 ; C2] = [A1 ; A2] * [B1 ; B2]
where A1 is a snpmatrix and A2 is a dense Matrix{Float}. Used for cleaner code.
Here we are separating the computation because A1 is stored in compressed form while A2 is
uncompressed (float64) matrix. This means that they cannot be stored in the same data
structure.
"""
function A_mul_B!(C1::AbstractMatrix{T}, C2::AbstractMatrix{T}, A1::SnpBitMatrix,
        A2::AbstractVecOrMat{T}, B1::AbstractVecOrMat{T}, B2::AbstractVecOrMat{T}) where {T <: Real}
		for i in 1:size(C1, 2)
			SnpArrays.mul!(C1[:, i], A1, B1[:, i])
		end
    LinearAlgebra.mul!(C2, A2, B2)
	C1 += C2
end

function A_mul_B!(C1::AbstractMatrix{T}, C2::AbstractMatrix{T}, A1::AbstractMatrix{T},
        A2::AbstractMatrix{T}, B1, B2) where {T <: Real}
    LinearAlgebra.mul!(C1, A1, B1)
    LinearAlgebra.mul!(C2, A2, B2)
	C1 += C2
end
