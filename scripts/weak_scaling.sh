inputs=(256 512 1024 2048 4096 8192 16384 32768)
kernel_dim=(4 8 16 32 64 128 256 512 1024)
iter=1000
ip=
program="1r-star2d" # [1r-star2d, 1r-box2d, star2d, box2d]
radius=1

for((i = 0; i < ${#inputs[@]}; i++)); do
  ./cstencil.sh $program real ip=$ip kernel_dim_x=${kernel_dim[$i]} kernel_dim_y=${kernel_dim[$i]} iterations=$iter\
                input_x=${input[$i]} input_y=${input[$i]} radius=$radius >> weak_scaling.csv
done
            