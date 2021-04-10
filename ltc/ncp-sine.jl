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


function lossf(x,y,m)
  Flux.reset!(m)
  ŷ = m.(x)
  sum([(ŷ[i][end,1] - y[i][1,1]) ^ 2 for i in 1:length(y)]), ŷ
end

function callback(x,y,c)
    l,pred = lossf(x, y, c)
    println(l)
    fig = plot([x[end,1] for x in pred])
    scatter!([x[1,1] for x in y])
    display(fig)
end

function loss_sciml(θ)
  m = re(θ)
  x, y = data_x, data_y
  ŷ = m.(x)
  sum(Flux.Losses.mse.(ŷ,y)), ŷ
end

function loss_galactic(θ,x0,re,x,y)
  m = re(θ)
  #x, y = data_x, data_y
  ŷ = m.(x)
  sum([(ŷ[i][end,1] - y[i][1,1]) ^ 2 for i in 1:length(y)]), ŷ
end


function cbs(θ,l,pred;doplot=true) #callback function to observe training
  display(l)
  if doplot
    fig = plot([x[end,1] for x in pred])
    #scatter!([x[1] for x in y])
    display(fig)
  end
  return false
end


function train()
    #ncp = NCPNet(2, 2,7,0,1, connections="full")
    #ncp = NCPNet(2, 2,3,4,1,2,2,3,4)
    ncp = NCPNet(2, 2,7,0,1,2,2,3,4)
    lower, upper = get_bounds(ncp.cell)
    chain = Chain(Mapper(2),ncp,Mapper(1))
    #c = Flux.Chain(Mapper(2),LTCNS(2,2),Mapper(2),LTC(2,7),Mapper(7),LTCNS(7,1),Mapper(1))
    c = chain
    ps,re = Flux.destructure(c)
    θ = Flux.params(c)

    @show length(ps)
    @show sum(length.(Flux.trainable(ncp.cell)))

    @show length(θ)
    @show sum(length.(θ))

    sens_mask = ncp.cell.sens_mask
    syn_mask = ncp.cell.syn_mask

    @show sum(sens_mask)
    @show sum(syn_mask)


    data_x, data_y = generate_data()

    @time loss_galactic(ps,ps,re,data_x,data_y)
    @time loss_galactic(ps,ps,re,data_x,data_y)
    @time loss_galactic(ps,ps,re,data_x,data_y)
    @time loss_galactic(ps,ps,re,data_x,data_y)


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
    # data = ncycle([(data_x, data_y)], 1000)
    Flux.train!((x,y) -> lossf(x,y,c)[1], θ, ncycle([(data_x, data_y)], 10), Flux.ADAM(0.001f0),cb=()->callback(data_x,data_y,c))
    Flux.train!((x,y) -> lossf(x,y,c)[1], θ, ncycle([(data_x, data_y)], 1000), Flux.ADAM(0.01f0),cb=()->callback(data_x,data_y,c))
    Flux.train!((x,y) -> lossf(x,y,c)[1], θ, ncycle([(data_x, data_y)], 100), Flux.ADAM(0.001f0),cb=()->callback(data_x,data_y,c))
    Flux.train!((x,y) -> lossf(x,y,c)[1], θ, ncycle([(data_x, data_y)], 1000), Flux.ADAM(0.001f0),cb=()->callback(data_x,data_y,c))
    Flux.train!((x,y) -> lossf(x,y,c)[1], θ, ncycle([(data_x, data_y)], 1000), Flux.ADAM(0.0001f0),cb=()->callback(data_x,data_y,c))




    f = OptimizationFunction((x,p)->loss_galactic(x,p,re,data_x,data_y),GalacticOptim.AutoZygote())
    prob = OptimizationProblem(f,ps,[0f0])
    sol = solve(prob, NelderMead(), maxiters=3, cb = cbs)
    @time sol = solve(prob,NelderMead(), maxiters=100, cb = cbs)
end
@time train()
