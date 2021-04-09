include("ltc-modelcleanup.jl")

function generate_data()
    in_features = 2
    out_features = 1
    N = 48
    data_x = [sin.(range(0,stop=3π,length=N)), cos.(range(0,stop=3π,length=N))]
    data_x = [reshape([data_x[1][i],data_x[2][i]],2,1) for i in 1:N] |> f32
    data_y = [reshape([y],1,1) for y in sin.(range(0,stop=6π,length=N))] |> f32

    data_x, data_y
end


function lossf(m, x,y)
  Flux.reset!(m)
  ŷ = m.(x)
  sum(Flux.Losses.mse.(ŷ,y))
end
callback(c,x,y) = println(lossf(c, x, y))

function loss_sciml(θ)
  m = re(θ)
  x, y = data_x, data_y
  ŷ = m.(x)
  sum(Flux.Losses.mse.(ŷ,y)), ŷ
end

function loss_galactic(θ,re,x,y)
  m = re(θ)
  #x, y = data_x, data_y
  ŷ = m.(x)
  sum(Flux.Losses.mse.(ŷ,y)), ŷ
end


function cbs(θ,l,pred;doplot=false) #callback function to observe training
  display(l)
  if doplot
    fig = plot([x[1] for x in pred])
    scatter!([x[1] for x in data_y])
    display(fig)
  end
  return false
end


function train()
    ncp = NCPNet(2, 2,3,4,1,2,2,3,4)
    #c = Flux.Chain(Mapper(2),LTCNS(2,2),Mapper(2),LTC(2,7),Mapper(7),LTCNS(7,1),Mapper(1))
    c = ncp
    ps,re = Flux.destructure(c)
    θ = Flux.params(c)

    lower, upper = get_bounds(ncp.cell)
    data_x, data_y = generate_data()

    @time loss_galactic(ps,re,data_x,data_y)
    @time loss_galactic(ps,re,data_x,data_y)
    @time loss_galactic(ps,re,data_x,data_y)
    @time loss_galactic(ps,re,data_x,data_y)


    fig = plot([x[1] for x in data_x])
    plot!(fig, [x[2] for x in data_x])
    plot!(fig, [x[1] for x in data_y])
    display(fig)

    # g = Zygote.gradient(Params(ps)) do
    #  loss_galactic(ps,re,data_x,data_y)[1]
    # end


    # f::F
    # u0::uType
    # p::P
    # lb::B
    # ub::B
    # lcons::LC
    # ucons::UC
    # kwargs::K

    data = ncycle([(data_x, data_y)], 10)
    Flux.train!((x,y) -> lossf(c,data_x,data_y), θ, data, Flux.ADAM(0.001f0),cb=()->callback(c,data_x,data_y))

    f = OptimizationFunction((p,x)->loss_galactic(ps,re,data_x,data_y),GalacticOptim.AutoZygote())
    prob = OptimizationProblem(f,ps)
    sol = solve(prob,ADAM(), maxiters=3, cb = cbs)
    #@time sol = solve(prob,NelderMead(), maxiters=1000, cb = cbs)
end
train()
