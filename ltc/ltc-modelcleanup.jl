using Flux
using OrdinaryDiffEq
using DiffEqSensitivity
using Distributions
using NPZ
using BenchmarkTools
#using GalacticOptim
using Plots
using IterTools: ncycle
#using ComponentArrays
using Parameters
using UnicodePlots

rand_uniform(lb,ub,dims...) = Float32.(rand(Uniform(lb,ub),dims...))# |> f32




struct Mapper{V}
  W::V
  b::V
  Mapper(W::V,b::V)  where {V} = new{typeof(W)}(W,b)
end
Mapper(in::Integer) = Mapper(ones(Float32,in), zeros(Float32,in))
(m::Mapper)(x::AbstractArray) = m.W .* x .+ m.b
Flux.@functor Mapper

function Base.show(io::IO, l::Mapper)
  print(io, "Mapper(", size(l.W,1))
  print(io, ")")
end



struct gNN{A}
  G::A
  μ::A
  σ::A
  gNN(G::A, μ::A, σ::A)  where {A} = new{typeof(G)}(G, μ, σ)
end
function gNN(in::Integer, out::Integer)
  G = rand_uniform(0.001,1, in,out)
  μ = rand_uniform(3,8, in,out)
  σ = rand_uniform(0.3,0.8, in,out)
  gNN(G,μ,σ)
end

function (m::gNN)(h::AbstractArray, x::AbstractArray)
  @views hc = [m.G[:,i][:] .* Flux.sigmoid.((x .- m.μ[:,i][:]) .* m.σ[:,i][:]) for i in 1:size(m.G,2)]
  hc, hc
end

Flux.@functor gNN
Flux.trainable(m::gNN) = (m.G, m.μ, m.σ,)

function get_bounds(m::gNN)
  lb = [
    [0 for _ in m.G]...,
    [-2 for _ in m.μ]...,
    [1e-3 for _ in m.σ]...,
  ]
  ub = [
    [1e3 for _ in m.G]...,
    [10 for _ in m.μ]...,
    [0.9 for _ in m.σ]...,
  ]
  lb, ub
end
function Base.show(io::IO, l::gNN)
  print(io, "gNN(", size(l.G,1), ", ", size(l.G, 2))
  print(io, ")")
end



struct NCP{I <: Integer, V, A, N1, F1 <: Function, N2, F2 <: Function, S}
  n_sensory::I # == in
  n_inter::I
  n_command::I
  n_motor::I # == out
  n_total::I

  cm::V
  Gleak::V
  Eleak::V

  sens_mask::A
  Esens::A
  fsens::N1
  psens::V
  resens::F1

  syn_mask::A
  Esyn::A
  fsyn::N2
  psyn::V
  resyn::F2

  state0::S
  prob#::PROB
  NCP(n_sensory,n_inter,n_command,n_motor,n_total,cm,Gleak,Eleak,sens_mask,Esens,fsens,psens,resens,syn_mask,Esyn,fsyn,psyn,resyn,state0,prob) = #new(cm,Gleak,Eleak,Esens,fsens,psens,resens,Esyn,fsyn,psyn,resyn,state0,prob)
    new{typeof(n_sensory),typeof(cm),typeof(Esens),typeof(fsens),typeof(resens),typeof(fsyn),typeof(resyn),typeof(state0)}(
      n_sensory,n_inter,n_command,n_motor,n_total,cm,Gleak,Eleak,sens_mask,Esens,fsens,psens,resens,syn_mask,Esyn,fsyn,psyn,resyn,state0,prob)

end

struct ODEParams
    I
    cm
    Gleak
    Eleak
    psens
    Esens
    psyn
    Esyn
end

function dnetdt!(dx,x,p,t, resens, sens_mask, resyn, syn_mask, sizes, lengths)
  #I::Vector{Float32}, cm::Array{Float32,2}, Gleak::Array{Float32,2}, Eleak::Array{Float32,2}, psens::Array{Float32,2}, Esens::Array{Float32,2}, psyn::Array{Float32,2}, Esyn::Array{Float32,2} = p
  #I, cm, Gleak, Eleak, psens, Esens, psyn, Esyn = p.I, p.cm, p.Gleak, p.Eleak, p.psens, p.Esens, p.psyn, p.Esyn
  #I = ica.I

  I_s     = 1
  I_e     = lengths[1]
  cm_s    = I_e + 1
  cm_e    = cm_s + lengths[2] - 1
  Gleak_s = cm_e + 1
  Gleak_e = Gleak_s + lengths[3] - 1
  Eleak_s = Gleak_e + 1
  Eleak_e = Eleak_s + lengths[4] - 1
  psens_s = Eleak_e + 1
  psens_e = psens_s + lengths[5] - 1
  Esens_s = psens_e + 1
  Esens_e = Esens_s + lengths[6] - 1
  psyn_s  = Esens_e + 1
  psyn_e  = psyn_s + lengths[7] - 1
  Esyn_s  = psyn_e + 1
  Esyn_e  = length(p)

  @views I     = p[reshape(I_s:I_e, sizes[1])]
  @views cm    = p[reshape(cm_s:cm_e, sizes[2])]
  @views Gleak = p[reshape(Gleak_s:Gleak_e, sizes[3])]
  @views Eleak = p[reshape(Eleak_s:Eleak_e, sizes[4])]
  @views psens = p[reshape(psens_s:psens_e, sizes[5])]
  @views Esens = p[reshape(Esens_s:Esens_e, sizes[6])]
  @views psyn  = p[reshape(psyn_s:psyn_e, sizes[7])]
  @views Esyn  = p[reshape(Esyn_s:Esyn_e, sizes[8])]



  fsens = resens(psens)(x,I)[2]
  @views fsens = reduce(hcat, fsens[i] for i in 1:size(fsens,1))#::Array{Float32,2}
  fsyn = resyn(psyn)(x,x)[2]
  @views fsyn = reduce(hcat, fsyn[i] for i in 1:size(fsyn,1))#::Array{Float32,2}

  fsens = fsens .* sens_mask
  fsyn = fsyn .* syn_mask

  #fsyn = hcat(resyn(psyn)(x,x)[2]...)

  @views I_sens = (sum(fsens[:,i] .* (Esens[:,i] .- x[i])) for i in 1:size(x,1))#::Array{Float32,1}
  @views I_syn = (sum(fsyn[:,i] .* (Esyn[:,i] .- x[i])) for i in 1:size(x,1))#::Array{Float32,1}


  dx .= (-1 ./ cm) .* (Gleak .* (Eleak .- x)) .+ I_sens .+ I_syn
  nothing
end

function NCP(in, n_sensory,n_inter,n_command,n_motor,sensory_out,inter_out,rec_command_out,motor_in)

  n_total = n_sensory + n_inter + n_command + n_motor
  sensory_s = 1
  inter_s = n_sensory + 1
  command_s = n_sensory + n_inter + 1
  motor_s = n_sensory + n_inter + n_command + 1

  sens_mask = zeros(Float32, in, n_total) # n_total +1 (extern) ?
  syn_mask  = zeros(Float32, n_total,n_total)

  sens_mask[:,1:n_sensory] .= 1f0

  for src in sensory_s:sensory_s+n_sensory-1
    for dst in collect(inter_s:inter_s+n_inter-1)
      syn_mask[src,dst] = 1f0
    end
  end
  for src in inter_s:inter_s+n_inter-1
    for dst in [collect(command_s:command_s+n_command-1); collect(inter_s:inter_s+n_inter-1)]
      syn_mask[src,dst] = 1f0
    end
  end
  for src in inter_s:inter_s+n_inter-1
    for dst in collect(motor_s:motor_s+n_motor-1)
      syn_mask[src,dst] = 1f0
    end
  end

  # for src in 1:n_sensory
  #   for dst in [rand(inter_s:inter_s+n_inter-1) for i in 1:sensory_out]
  #     syn_mask[dst,src] = 1f0
  #   end
  # end
  # for src in 1:n_inter
  #   for dst in [rand(command_s:command_s+n_command-1) for i in 1:inter_out]
  #     syn_mask[dst,src] = 1f0
  #   end
  # end
  # for src in 1:n_command
  #   for dst in [rand(command_s:command_s+n_command-1) for i in 1:rec_command_out]
  #     syn_mask[dst,src] = 1f0
  #   end
  # end
  # for dst in 1:n_motor
  #   for src in [rand(command_s:command_s+n_command-1) for i in 1:motor_in]
  #     syn_mask[dst,src] = 1f0
  #   end
  # end

  display(Plots.heatmap(sens_mask))
  display(Plots.heatmap(syn_mask))

  cm        = rand_uniform(0.4,0.6, n_total)
  Gleak     = rand_uniform(0.001,1, n_total)
  Eleak     = reshape([[1f0,-1f0][rand(1:2)] for i in 1:n_total], n_total)#rand_uniform(-1,1, out) #rand_uniform(-0.2,0.2, out)

  Esens     = reshape([[1f0,-1f0][rand(1:2)] for i in 1:in*n_total], in,n_total)# rand_uniform(-1,1, in, out) #randn(Float32, in, out)
  Esyn      = reshape([[1f0,-1f0][rand(1:2)] for i in 1:n_total*n_total], n_total,n_total)#randn(Float32, out, out)

  nnsens = gNN(in,n_total)
  psens, resens    = Flux.destructure(nnsens)

  nnsyn = gNN(n_total,n_total)
  psyn, resyn    = Flux.destructure(nnsyn)

  state0    = zeros(Float32, n_total,1)
  prob = ODEProblem((dx,x,p,t)->dnetdt!(dx,x,p,t,resens, sens_mask, resyn, syn_mask),zeros(Float32,size(state0,1)),(0f0,1f0))

  NCP(n_sensory,n_inter,n_command,n_motor,n_total,cm,Gleak,Eleak,sens_mask,Esens,nnsens,psens,resens,syn_mask,Esyn,nnsyn,psyn,resyn,state0,prob)
end

(m::NCP)(h::AbstractVector, x::AbstractArray) = m(repeat(h,size(x,2))) #m(Flux.stack([h[:,1] for _ in 1:size(x,2)], 2), x)


function (m::NCP)(h::AbstractMatrix, x::AbstractArray)
  cm, Gleak, Eleak, psens, Esens, psyn, Esyn = m.cm, m.Gleak, m.Eleak, m.psens, m.Esens, m.psyn, m.Esyn

  hs = h

  # function prob_func(prob,i,repeat)
  #   #f = (du,u,p,t) -> dudt!(du,u,p,t,i)
  #   p = [x, cm, Gleak, Eleak, psens, Esens, psyn, Esyn]
  #   sizes = size.(p)
  #   lengths = prod.(sizes)
  #   @views p = [x[:]; cm[:]; Gleak[:]; Eleak[:]; psens[:]; Esens[:]; psyn[:]; Esyn[:]]
  #   f = ODEFunction((dx,x,p,t) -> dnetdt!(dx,x,p,t,m.resens, m.sens_mask, m.resyn, m.syn_mask, sizes, lengths))
  #   @vies hi = h[:,i]
  #   remake(m.prob, f=f, u0=hi, p=p)
  #   #ODEProblem((du,u,p,t) -> dudt!(du,u,p,t,i),u0[:,1],(T1,T2),p)
  # end
  # function output_func(sol,i)
  #   last(sol), false
  # end
  # ensemble_prob = EnsembleProblem(m.prob,prob_func=prob_func, output_func=output_func, safetycopy=false)
  # sol = solve(ensemble_prob,VCABM(),EnsembleThreads(),trajectories=size(x,2))
  # return h,sol
  #ica = ComponentArray(I=x[:])
  #p = ComponentArray(cm=cm, Gleak=Gleak, Eleak=Eleak, psens=psens, Esens=Esens, psyn=psyn, Esyn=Esyn)



  #p = ODEParams(x,cm,Gleak,Eleak,psens,Esens,psyn,Esyn)
  #p = [x, cm, Gleak, Eleak, psens, Esens, psyn, Esyn]
  sizes = [size(x),size(cm),size(Gleak),size(Eleak),size(psens),size(Esens),size(psyn),size(Esyn)]
  lengths = [prod(sizes[1]),prod(sizes[2]),prod(sizes[3]),prod(sizes[4]),prod(sizes[5]),prod(sizes[6]),prod(sizes[7]),prod(sizes[8])]
  @views p = [x[:]; cm[:]; Gleak[:]; Eleak[:]; psens[:]; Esens[:]; psyn[:]; Esyn[:]]
  #tmpprob = remake(m.prob, p=p, u0=hs)
  #tmpprob = m.prob
  tmpprob = ODEProblem((dx,x,p,t)->dnetdt!(dx,x,p,t,m.resens, m.sens_mask, m.resyn, m.syn_mask, sizes, lengths),h,(0f0,1f0),p)
  sol = Array(solve(tmpprob,VCABM(),sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))#::Array{Float32,3}
  #@show size(sol)
  @views sol = sol[:,:,end]

  return h, sol

end


NCPNet(args...) = Flux.Recur(NCP(args...))
Flux.Recur(m::NCP) = Flux.Recur(m, m.state0)
Flux.@functor NCP #(cm, Gleak, Eleak, sens_mask, Esens, psens, syn_mask, Esyn, psyn, state0)
Flux.trainable(m::NCP) = (m.cm, m.Gleak, m.Eleak, m.Esens, m.psens, m.Esyn, m.psyn, m.state0,)

function get_bounds(m::NCP)
  lower = [
      [0 for _ in m.cm]...,
      [0 for _ in m.Gleak]...,
      [-1.1 for _ in m.Eleak]...,
      [-1.1 for _ in m.Esens]...,
      get_bounds(m.fsens)[1]...,
      [-1.1 for _ in m.Esyn]...,
      get_bounds(m.fsyn)[1]...,
      [-2 for _ in m.state0]...,
  ] |> f32
  upper = [
    [1 for _ in m.cm]...,
    [1 for _ in m.Gleak]...,
    [1.1 for _ in m.Eleak]...,
    [1.1 for _ in m.Esens]...,
    get_bounds(m.fsens)[2]...,
    [1.1 for _ in m.Esyn]...,
    get_bounds(m.fsyn)[2]...,
    [2 for _ in m.state0]...,
  ] |> f32

  lower, upper
end
