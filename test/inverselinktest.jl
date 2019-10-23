import TraitSimulation: cauchit_inverse_link, cloglog_inverse_link, inverse_inverse_link, identity_inverse_link, logit_inverse_link, log_inverse_link, probit_inverse_link

float64_input = Float64[0, 1]
float64_output = Float64[0.5, 0.75]

for i in eachindex(float64_input)
  return(@test cauchit_inverse_link(float64_input[i]) == float64_output[i])
end

float32_input = Float32[0, 1]
float32_output = Float32[0.5, 0.75]
@test cauchit_inverse_link(float32_input[1]) isa Float32

# #And update our algorithm: 
# cauchit_inverse_link(x) = atan(x) / pi + one(x) / 2

for i in eachindex(float32_input)
  return(@test cauchit_inverse_link(float32_input[i]) == float32_output[i])
end


# cloglog
float64_input = Float64[0, 1]
float64_output = Float64[0.6321205588285577, 0.9340119641546875]

for i in eachindex(float64_input)
  return(@test cloglog_inverse_link(float64_input[i]) == float64_output[i])
end

float32_input = Float32[0, 1]
float32_output = Float32[0.6321205588285577, 0.9340119641546875]
@test cloglog_inverse_link(float32_input[1]) isa Float32

# #And update our algorithm: 
# function cloglog_inverse_link(x)
#   return one(x) - exp(-exp(x))
# end 

for i in eachindex(float32_input)
  return(@test cloglog_inverse_link(float32_input[i]) == float32_output[i])
end

# identity
float64_input = Float64[0, 1]
float64_output = Float64[0.0, 1.0]

for i in eachindex(float64_input)
  return(@test identity_inverse_link(float64_input[i]) == float64_output[i])
end

float32_input = Float32[0, 1]
float32_output = Float32[0.0, 1.0]
@test identity_inverse_link(float32_input[1]) isa Float32

# #And update our algorithm: 
# function identity_inverse_link(x)
#   return x
# end

for i in eachindex(float32_input)
  return(@test identity_inverse_link(float32_input[i]) == float32_output[i])
end

# inverse inverse 
float64_input = Float64[1, 2]
float64_output = Float64[1.0, 2.0]

for i in eachindex(float64_input)
  return(@test inverse_inverse_link(float64_input[i]) == float64_output[i])
end

float32_input = Float32[1, 2]
float32_output = Float32[1.0, 2.0]
@test inverse_inverse_link(float32_input[1]) isa Float32

# #And update our algorithm: 
# function inverse_inverse_link(x)
#   return one(x) / x
# end

for i in eachindex(float32_input)
  return(@test inverse_inverse_link(float32_input[i]) == float32_output[i])
end

# logit link
float64_input = Float64[0, 1]
float64_output = Float64[0.5, 0.7310585786300049]

for i in eachindex(float64_input)
  return(@test logit_inverse_link(float64_input[i]) == float64_output[i])
end

float32_input = Float32[0, 1]
float32_output = Float32[0.5, 0.7310585786300049]
@test logit_inverse_link(float32_input[1]) isa Float32


for i in eachindex(float32_input)
  return(@test logit_inverse_link(float32_input[i]) == float32_output[i])
end

# log link
float64_input = Float64[0, 1]
float64_output = Float64[1.0, 2.718281828459045]

for i in eachindex(float64_input)
  return(@test log_inverse_link(float64_input[i]) == float64_output[i])
end

float32_input = Float32[0, 1]
float32_output = Float32[1.0, 2.718281828459045]
@test log_inverse_link(float32_input[1]) isa Float32


for i in eachindex(float32_input)
  return(@test log_inverse_link(float32_input[i]) == float32_output[i])
end

# probit 

float64_input = Float64[0, 1]
float64_output = Float64[0.5, 0.8413447460685429]

for i in eachindex(float64_input)
  return(@test probit_inverse_link(float64_input[i]) == float64_output[i])
end

float32_input = Float32[0, 1]
float32_output = Float32[0.5, 0.8413447460685429]
@test probit_inverse_link(float32_input[1]) isa Float32


for i in eachindex(float32_input)
  return(@test probit_inverse_link(float32_input[i]) == float32_output[i])
end